import torch
import os
import argparse
import sys
from typing import Optional
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import torch.nn.functional as F

from main_code.model.gen_model import GenModel
from main_code.model.ref_model import RefModel
from main_code.model.de_net import DEResNet
from main_code.model.anymate_anyone.reference_attention import ReferenceNetAttention
from visualize_aicg import AICGVisualizer
from aicg_steering import AICGSteerer
from face_preproc import (
    compute_dynamic_sigma,
    soften_mask,
    degrade_lr_tile,
    bbox_max_size_from_mask,
)

from ram.models.ram import ram
from ram.models.ram_lora import ram as ram_deg
from ram import inference_ram as inference

from my_utils.wavelet_color import wavelet_color_fix
from accelerate import Accelerator
from omegaconf import OmegaConf
from my_utils.testing_utils import parse_args_paired_testing


ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}


# ── Tiling utilities ────────────────────────────────────────────────────────

def _1d_blend(size, overlap, at_start, at_end):
    """Linear ramp weights for one spatial dimension.

    At image borders (at_start / at_end) weight stays 1 so that edge
    pixels are never down-weighted without a neighbouring tile to compensate.
    """
    w = torch.ones(size, dtype=torch.float32)
    if not at_start and overlap > 0:
        w[:overlap] = torch.linspace(0.0, 1.0, overlap)
    if not at_end and overlap > 0:
        w[size - overlap:] = torch.linspace(1.0, 0.0, overlap)
    return w


def tile_weight_map(tile_size, overlap, ty, tx, img_h, img_w):
    """2-D blending weight for the tile starting at (ty, tx)."""
    wy = _1d_blend(tile_size, overlap, ty == 0, ty + tile_size >= img_h)
    wx = _1d_blend(tile_size, overlap, tx == 0, tx + tile_size >= img_w)
    return wy.unsqueeze(1) * wx.unsqueeze(0)   # [tile_size, tile_size]


def tile_start_coords(total, tile_size, overlap):
    """1-D tile start positions that cover [0, total) completely."""
    if total <= tile_size:
        return [0]
    stride = tile_size - overlap
    coords = list(range(0, total - tile_size, stride))
    last = max(0, total - tile_size)
    if not coords or coords[-1] != last:
        coords.append(last)
    return coords


# ── Visualization path helper ────────────────────────────────────────────────

def _resolve_vis_path(vis_output_path: str | None, sr_output_path: str) -> str:
    """[Fix 1] vis_output_path 가 폴더이거나 None 일 때 파일 경로를 확정.

    - None        → sr_output_path 와 같은 디렉터리에 <stem>_aicg_vis.png
    - 폴더 경로   → 그 폴더 안에 <stem>_aicg_vis.png
    - 파일 경로   → 그대로 사용
    """
    stem = os.path.splitext(os.path.basename(sr_output_path))[0]
    fname = stem + "_aicg_vis.png"

    if not vis_output_path:
        return os.path.join(os.path.dirname(os.path.abspath(sr_output_path)), fname)
    if os.path.isdir(vis_output_path):
        return os.path.join(vis_output_path, fname)
    return vis_output_path


# ── Image collection ─────────────────────────────────────────────────────────

def collect_image_files(folder):
    """Return sorted list of image filenames in folder (extension filter applied)."""
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )
    return files


def find_roi_by_stem(roi_folder, stem):
    """roi_folder 안에서 확장자 무관하게 stem 이 일치하는 첫 번째 이미지 경로를 반환.

    예) stem='img001', roi_folder에 'img001.png' 존재 → 해당 경로 반환.
        일치하는 파일이 없으면 None 반환.
    """
    for fname in os.listdir(roi_folder):
        s, ext = os.path.splitext(fname)
        if s == stem and ext.lower() in SUPPORTED_EXTENSIONS:
            return os.path.join(roi_folder, fname)
    return None


# ── Main inference ───────────────────────────────────────────────────────────

def load_models(args, device, weight_dtype):
    net_sr = GenModel(
        sd_path=args.sd_path,
        pretrained_backbone_path=args.pretrained_backbone_path,
        pretrained_ref_gen_path=args.pretrained_ref_gen_path,
        args=args,
    )
    net_ref = RefModel(sd_path=args.sd_path)
    net_de  = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)

    model_vlm = ram(
        pretrained=args.ram_path, image_size=384, vit='swin_l'
    ).eval().to(device, dtype=torch.float16)
    model_vlm_deg = ram_deg(
        pretrained=args.ram_path, pretrained_condition=args.dape_path,
        image_size=384, vit='swin_l'
    ).eval().to(device, dtype=torch.float16)

    ref_writer = ReferenceNetAttention(
        net_ref.unet, mode='write', fusion_blocks=args.fusion_blocks,
        is_image=True, dtype=weight_dtype,
    )
    ref_reader = ReferenceNetAttention(
        net_sr.unet, mode='read', fusion_blocks=args.fusion_blocks,
        is_image=True, dtype=weight_dtype,
    )

    net_sr.to(device, dtype=weight_dtype).eval()
    net_ref.to(device, dtype=weight_dtype).eval()
    net_de.to(device, dtype=weight_dtype).eval()

    return net_sr, net_ref, net_de, model_vlm, model_vlm_deg, ref_writer, ref_reader


def infer_batch(net_sr, net_ref, net_de, ref_writer, ref_reader,
                x_src_b, x_ref_b, prompts_src, prompts_ref, weight_dtype):
    """Run one batched SR forward pass. Returns raw model output [B,3,H,W] in [-1,1]."""
    with torch.no_grad():
        deg_scores  = net_de(x_src_b)
        net_ref(x_ref_b * 2.0 - 1.0, prompt=prompts_ref)
        ref_reader.update(ref_writer, dtype=weight_dtype)
        predictions = net_sr(x_src_b * 2.0 - 1.0, deg_scores, prompt=prompts_src)
        ref_reader.clear()
        ref_writer.clear()
    return predictions


def _infer_single_image(lq_path, ref_path, output_path,
                        net_sr, net_ref, net_de, model_vlm, model_vlm_deg,
                        ref_writer, ref_reader, args, device, weight_dtype,
                        tile_size, overlap, batch_sz, scale,
                        visualize=False, roi_path=None, vis_output_path=None):
    """Process a single (lq, ref) image pair and save the result.

    LQ is bicubic-upscaled by `scale` before tiling so the network output
    covers `scale`× the original LQ resolution.  Ref is kept at its original
    resolution and tiled proportionally (no pre-upscaling).
    """

    # ── Load & align original images ─────────────────────────────────────────
    # exif_transpose: JPEG EXIF 회전 태그를 픽셀에 실제 적용.
    # PIL은 기본적으로 EXIF를 무시해 Windows 뷰어와 방향이 달라질 수 있음.
    lq_img  = ImageOps.exif_transpose(Image.open(lq_path).convert("RGB"))
    ref_img = ImageOps.exif_transpose(Image.open(ref_path).convert("RGB"))

    # Original LQ size aligned to multiples of 8 (VAE requirement)
    orig_w = lq_img.size[0] // 8 * 8
    orig_h = lq_img.size[1] // 8 * 8
    lq_img = lq_img.resize((orig_w, orig_h), Image.BICUBIC)

    ref_img = ref_img.resize(
        (ref_img.size[0] // 8 * 8, ref_img.size[1] // 8 * 8), Image.BICUBIC
    )

    to_tensor = transforms.ToTensor()
    x_lq_orig = to_tensor(lq_img)   # [3, H,  W ] – original LQ (for RAM prompt)
    x_ref      = to_tensor(ref_img)  # [3, Hr, Wr] – ref kept at original resolution

    ref_h, ref_w = x_ref.shape[1], x_ref.shape[2]

    # ── Bicubic upscale LQ by `scale` (ref is NOT upscaled) ──────────────────
    sr_h = orig_h * scale // 8 * 8   # align SR canvas to multiples of 8
    sr_w = orig_w * scale // 8 * 8
    x_lq = F.interpolate(
        x_lq_orig.unsqueeze(0),
        size=(sr_h, sr_w),
        mode='bicubic', align_corners=False,
    ).squeeze(0).clamp(0, 1)          # [3, H*scale, W*scale]

    lq_h, lq_w = x_lq.shape[1], x_lq.shape[2]

    # Bicubic LQ (PIL) used as wavelet color reference – must match SR output size
    lq_bicubic_img = transforms.ToPILImage()(x_lq.cpu())

    print(f"  LQ: {orig_w}x{orig_h}  ->  SR canvas: {lq_w}x{lq_h}  (x{scale})")

    # ── Global semantic prompts (computed from original-resolution images) ────
    # inference_ram() returns List[str]; extract [0] to keep prompt as plain str.
    with torch.no_grad():
        prompt_ref = inference(
            ram_transforms(x_ref.unsqueeze(0)).to(device, dtype=torch.float16),
            model_vlm)[0]
        prompt_src = inference(
            ram_transforms(x_lq_orig.unsqueeze(0)).to(device, dtype=torch.float16),
            model_vlm_deg)[0]
    print(f"  Prompt (ref): {prompt_ref}")
    print(f"  Prompt (src): {prompt_src}")

    # ── ROI 마스크 로드 (visualize / face_preproc / steering 공통 사용) ──────
    roi_mask = None
    if roi_path and os.path.exists(roi_path):
        roi_pil  = ImageOps.exif_transpose(Image.open(roi_path).convert("L"))
        roi_pil  = roi_pil.resize((lq_w, lq_h), Image.BILINEAR)
        roi_mask = np.array(roi_pil).astype(np.float32) / 255.0  # [lq_h, lq_w]
        print(f"  ROI mask loaded: {roi_path}")

    # ── Face preprocessing / AICG steering 설정 로드 ─────────────────────────
    enable_steering = bool(args.get("enable_steering", False))
    enable_preproc  = bool(args.get("enable_face_preproc", False))

    aicg_scale         = float(args.get("aicg_scale",        1.0))
    aicg_trust_scale   = float(args.get("aicg_trust_scale",  1.0))
    aicg_verify_scale  = float(args.get("aicg_verify_scale", 1.0))
    aicg_force_verify  = bool(args.get("aicg_force_verify",  False))

    face_sigma_ratio   = float(args.get("face_sigma_ratio",  0.02))
    face_sigma_min     = float(args.get("face_sigma_min",    1.5))
    face_sigma_max     = float(args.get("face_sigma_max",    8.0))
    face_noise_std     = float(args.get("face_noise_std",    0.08))
    face_blend_ratio   = float(args.get("face_blend_ratio",  0.6))
    face_soft_mask_on  = bool(args.get("face_soft_mask",     True))

    save_degraded_lr   = bool(args.get("save_degraded_lr",   False))
    degraded_lr_suffix = str(args.get("degraded_lr_suffix",  "_degrad_lr"))

    # Degraded LR output path: <output_dir>/<stem><suffix>.png
    _degrad_stem = os.path.splitext(os.path.basename(output_path))[0]
    degrad_out   = os.path.join(
        os.path.dirname(os.path.abspath(output_path)),
        _degrad_stem + degraded_lr_suffix + ".png",
    )

    # Face box 의 원본(=SR canvas) 해상도 픽셀 크기 → dynamic sigma 산출
    # (Tile 단위가 아닌 원본에서 한 번만 계산해 모든 tile 에 일관되게 적용)
    face_box_max_px = bbox_max_size_from_mask(roi_mask) if roi_mask is not None else 0
    dyn_sigma = compute_dynamic_sigma(
        face_box_max_px, face_sigma_ratio, face_sigma_min, face_sigma_max,
    ) if face_box_max_px > 0 else 0.0

    soft_mask_global: Optional[torch.Tensor] = None
    if roi_mask is not None and (enable_steering or enable_preproc):
        hard_mask_t = torch.from_numpy(roi_mask).float()
        if face_soft_mask_on and dyn_sigma > 0:
            soft_mask_global = soften_mask(hard_mask_t, dyn_sigma)
        else:
            soft_mask_global = hard_mask_t.clamp(0, 1)
        print(f"  Face box: max={face_box_max_px}px,  dyn_sigma={dyn_sigma:.2f},  "
              f"soft_mask={'on' if face_soft_mask_on else 'off'},  "
              f"steering={'on' if enable_steering else 'off'},  "
              f"preproc={'on' if enable_preproc else 'off'}")

    # AICGSteerer 초기화 (활성화된 경우에만)
    steerer: Optional[AICGSteerer] = None
    if enable_steering and soft_mask_global is not None:
        steerer = AICGSteerer(
            net_sr.unet,
            scale=aicg_scale,
            trust_scale=aicg_trust_scale,
            verify_scale=aicg_verify_scale,
            force_verify=aicg_force_verify,
            fusion_blocks=args.get("fusion_blocks", "full"),
        )
        print(f"  AICGSteerer: trust_eff={steerer.effective_trust_scale:.3f}, "
              f"verify_eff={steerer.effective_verify_scale:.3f}")

    # ── Single-tile fast-path (upscaled LQ fits in one tile) ─────────────────
    if lq_h <= tile_size and lq_w <= tile_size:
        print("  SR canvas fits in one tile – running single inference.")

        # 단일 타일에서도 face_preproc 동일하게 적용
        x_src_processed = x_lq
        if enable_preproc and soft_mask_global is not None and dyn_sigma > 0:
            x_src_processed = degrade_lr_tile(
                lq_tile_01=x_lq,
                soft_mask=soft_mask_global,
                sigma=dyn_sigma,
                noise_std=face_noise_std,
                degrad_blend_ratio=face_blend_ratio,
            )

        # Degraded LR 저장 (enable_preproc 가 True 이고 실제로 degradation 이 적용된 경우)
        if save_degraded_lr and enable_preproc and soft_mask_global is not None and dyn_sigma > 0:
            degrad_img = transforms.ToPILImage()(x_src_processed.clamp(0, 1).cpu())
            degrad_img.save(degrad_out)
            print(f"  Saved degraded LR: {degrad_out}")

        x_src_t = x_src_processed.unsqueeze(0).to(device, dtype=weight_dtype)
        x_ref_t = x_ref.unsqueeze(0).to(device, dtype=weight_dtype)

        # 단일 타일에서도 steering 적용 (whole-image face mask 사용)
        if steerer is not None and soft_mask_global is not None:
            steerer.set_batch_face_masks([soft_mask_global])

        def _run_infer():
            return infer_batch(
                net_sr, net_ref, net_de, ref_writer, ref_reader,
                x_src_t, x_ref_t, [prompt_src], [prompt_ref], weight_dtype,
            )

        if visualize:
            # [Fix 1] 폴더 경로·None 모두 파일 경로로 변환
            _vis_output = _resolve_vis_path(vis_output_path, output_path)
            # steerer 가 있으면 probs/gate 캐싱 활성화 → viz 가 개입 결과를 반영
            if steerer is not None:
                steerer.capture_for_viz = True
            viz = AICGVisualizer(
                net_sr.unet, roi_mask=roi_mask,
                ref_h=ref_h, ref_w=ref_w,
                lq_h=lq_h, lq_w=lq_w,
                tile_size=tile_size,
                fusion_blocks=args.get("fusion_blocks", "full"),
                steerer=steerer,
            )
            with viz.capture_tile(0, 0, 0, 0, ref_h, ref_w, lq_h, lq_w):
                if steerer is not None:
                    with steerer.apply_steering():
                        preds = _run_infer()
                else:
                    preds = _run_infer()
            viz.finalize(np.array(ref_img), np.array(lq_bicubic_img), _vis_output)
        else:
            if steerer is not None:
                with steerer.apply_steering():
                    preds = _run_infer()
            else:
                preds = _run_infer()

        if steerer is not None:
            steerer.clear_batch_face_masks()

        pred_img = transforms.ToPILImage()((preds[0] * 0.5 + 0.5).clamp(0, 1).cpu())
        if args.get('align_method', 'wavelet') == 'wavelet':
            pred_img = wavelet_color_fix(pred_img, lq_bicubic_img)
        pred_img.resize((orig_w * scale, orig_h * scale), Image.BICUBIC).save(output_path)
        print(f"  Saved: {output_path}")
        return

    # ── Build tile grid on the upscaled LQ canvas ────────────────────────────
    ys = tile_start_coords(lq_h, tile_size, overlap)
    xs = tile_start_coords(lq_w, tile_size, overlap)
    all_tiles = [(y, x) for y in ys for x in xs]
    n_tiles    = len(all_tiles)
    print(f"  Grid: {lq_h}x{lq_w} -> {len(ys)}x{len(xs)} = {n_tiles} tiles "
          f"(tile={tile_size}, overlap={overlap}, batch={batch_sz})")

    # Output accumulators at the SR canvas size (float32 for precision)
    pred_acc   = torch.zeros(3, lq_h, lq_w, dtype=torch.float32, device=device)
    weight_acc = torch.zeros(1, lq_h, lq_w, dtype=torch.float32, device=device)

    # Degraded LR accumulators (CPU, only allocated when needed)
    degrad_acc: Optional[torch.Tensor] = None
    degrad_wt:  Optional[torch.Tensor] = None
    if save_degraded_lr and enable_preproc:
        degrad_acc = torch.zeros(3, lq_h, lq_w, dtype=torch.float32)
        degrad_wt  = torch.zeros(1, lq_h, lq_w, dtype=torch.float32)

    # ── 시각화 캔버스 초기화 ─────────────────────────────────────────────────
    viz = None
    if visualize:
        # [Fix 1] 폴더 경로·None 모두 파일 경로로 변환
        _vis_output = _resolve_vis_path(vis_output_path, output_path)
        # steerer 가 있으면 probs/gate 캐싱 활성화 → viz 가 개입 결과를 반영
        if steerer is not None:
            steerer.capture_for_viz = True
        viz = AICGVisualizer(
            net_sr.unet, roi_mask=roi_mask,
            ref_h=ref_h, ref_w=ref_w,
            lq_h=lq_h, lq_w=lq_w,
            tile_size=tile_size,
            fusion_blocks=args.get("fusion_blocks", "full"),
            steerer=steerer,
        )

    # ── Tile inference loop ───────────────────────────────────────────────────
    def _crop_pad_mask(mask: torch.Tensor, ty: int, tx: int) -> torch.Tensor:
        """soft_mask_global 에서 [tile_size, tile_size] 크기로 crop+pad."""
        crop = mask[ty:ty + tile_size, tx:tx + tile_size]
        if crop.shape[0] == tile_size and crop.shape[1] == tile_size:
            return crop
        padded = torch.zeros(tile_size, tile_size, dtype=crop.dtype, device=crop.device)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        return padded

    for batch_start in range(0, n_tiles, batch_sz):
        batch_pos = all_tiles[batch_start : batch_start + batch_sz]
        B = len(batch_pos)
        print(f"    tiles {batch_start + 1}-{batch_start + B} / {n_tiles}")

        lq_tiles    = []
        ref_tiles   = []
        tile_coords = []   # (ty, tx, ry, rx, rth, rtw) per tile in batch
        face_masks_per_tile: list = []

        for (ty, tx) in batch_pos:
            # LQ tile: cropped from the bicubic-upscaled LQ
            lq_tile = x_lq[:, ty:ty + tile_size, tx:tx + tile_size]

            # Per-tile face mask (soft) – preproc / steering 양쪽에서 사용
            face_tile_mask = None
            if soft_mask_global is not None:
                face_tile_mask = _crop_pad_mask(soft_mask_global, ty, tx)
                # tile 영역에 얼굴 픽셀이 거의 없으면 None 으로 빠른 패스
                if not face_tile_mask.any():
                    face_tile_mask = None

            # 픽셀 공간 LR degradation (원본 해상도 기준 dyn_sigma 사용 → blur 일관성)
            if (enable_preproc and face_tile_mask is not None
                    and dyn_sigma > 0):
                # tile 외 영역은 zero-pad 됐을 수 있음 → 실제 LQ 영역 크기에 맞춰 crop
                th_actual = lq_tile.shape[1]
                tw_actual = lq_tile.shape[2]
                m_for_deg = face_tile_mask[:th_actual, :tw_actual]
                lq_tile = degrade_lr_tile(
                    lq_tile_01=lq_tile,
                    soft_mask=m_for_deg,
                    sigma=dyn_sigma,
                    noise_std=face_noise_std,
                    degrad_blend_ratio=face_blend_ratio,
                )
            lq_tiles.append(lq_tile)
            face_masks_per_tile.append(face_tile_mask)

            # Ref tile: proportionally cropped from the ORIGINAL ref (no upscaling).
            ry  = int(ty  * ref_h / lq_h)
            rx  = int(tx  * ref_w / lq_w)
            rth = max(1, round(tile_size * ref_h / lq_h))
            rtw = max(1, round(tile_size * ref_w / lq_w))
            ry  = min(ry, max(0, ref_h - rth))
            rx  = min(rx, max(0, ref_w - rtw))
            ref_crop = x_ref[:, ry : ry + rth, rx : rx + rtw]
            if rth != tile_size or rtw != tile_size:
                ref_crop = F.interpolate(
                    ref_crop.unsqueeze(0),
                    size=(tile_size, tile_size),
                    mode='bicubic', align_corners=False,
                ).squeeze(0)
            ref_tiles.append(ref_crop)
            tile_coords.append((ty, tx, ry, rx, rth, rtw))

        x_src_b = torch.stack(lq_tiles).to(device, dtype=weight_dtype)
        x_ref_b = torch.stack(ref_tiles).to(device, dtype=weight_dtype)

        prompts_ref_b = [prompt_ref] * B
        prompts_src_b = [prompt_src] * B

        # AICGSteerer 에 batch face masks 등록 (steering 컨텍스트 안에서만 효력)
        if steerer is not None:
            steerer.set_batch_face_masks(face_masks_per_tile)

        def _run_batch():
            return infer_batch(
                net_sr, net_ref, net_de, ref_writer, ref_reader,
                x_src_b, x_ref_b, prompts_src_b, prompts_ref_b, weight_dtype,
            )

        # viz / steering 이중 컨텍스트
        if viz is not None and steerer is not None:
            with viz.capture_batch(tile_coords, lq_h, lq_w), steerer.apply_steering():
                predictions = _run_batch()
        elif viz is not None:
            with viz.capture_batch(tile_coords, lq_h, lq_w):
                predictions = _run_batch()
        elif steerer is not None:
            with steerer.apply_steering():
                predictions = _run_batch()
        else:
            predictions = _run_batch()

        if steerer is not None:
            steerer.clear_batch_face_masks()

        for k, (ty, tx) in enumerate(batch_pos):
            w    = tile_weight_map(tile_size, overlap, ty, tx, lq_h, lq_w).to(device)
            pred = (predictions[k] * 0.5 + 0.5).clamp(0, 1).float()
            pred_acc[:,   ty:ty + tile_size, tx:tx + tile_size] += pred * w
            weight_acc[:, ty:ty + tile_size, tx:tx + tile_size] += w

            if degrad_acc is not None:
                w_cpu = tile_weight_map(tile_size, overlap, ty, tx, lq_h, lq_w)
                dt    = lq_tiles[k].float().cpu().clamp(0, 1)
                degrad_acc[:, ty:ty + tile_size, tx:tx + tile_size] += dt * w_cpu
                degrad_wt[:,  ty:ty + tile_size, tx:tx + tile_size] += w_cpu

    # ── Merge tiles & save ────────────────────────────────────────────────────
    result     = (pred_acc / weight_acc.clamp(min=1e-6)).clamp(0, 1)
    result_img = transforms.ToPILImage()(result.cpu())
    if args.get('align_method', 'wavelet') == 'wavelet':
        result_img = wavelet_color_fix(result_img, lq_bicubic_img)
    result_img.resize((orig_w * scale, orig_h * scale), Image.BICUBIC).save(output_path)
    print(f"  Saved: {output_path}  ({orig_w * scale}x{orig_h * scale})")

    # ── Degraded LR 저장 (multi-tile) ────────────────────────────────────────
    if degrad_acc is not None:
        degrad_merged = (degrad_acc / degrad_wt.clamp(min=1e-6)).clamp(0, 1)
        transforms.ToPILImage()(degrad_merged).save(degrad_out)
        print(f"  Saved degraded LR: {degrad_out}")

    # ── AICG 시각화 저장 ─────────────────────────────────────────────────────
    if viz is not None:
        viz.finalize(np.array(ref_img), np.array(lq_bicubic_img), _vis_output)


def run_demo_tiled(lq_path, ref_path, output_path, args):
    """
    폴더 경로가 주어지면 lq_path / ref_path 안의 동일 파일명 이미지를 전부 처리.
    파일 경로가 주어지면 단일 이미지 처리 (기존 동작).

    output_path:
      - 단일 모드: 저장할 파일 경로 (예: ./result.png)
      - 폴더 모드: 결과를 저장할 폴더 경로 (없으면 자동 생성)
    """
    accelerator  = Accelerator(mixed_precision=args.mixed_precision)
    device       = accelerator.device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    tile_size  = int(args.get("tile_size",       512))
    overlap    = int(args.get("tile_overlap",     64))
    batch_sz   = int(args.get("tile_batch_size",   4))
    scale      = int(args.get("scale",             4))
    visualize  = bool(args.get("visualize",      False))
    roi_path   = args.get("roi_path",            None)
    vis_out    = args.get("vis_output_path",     None)

    lq_is_dir  = os.path.isdir(lq_path)
    ref_is_dir = os.path.isdir(ref_path)

    roi_is_dir = os.path.isdir(roi_path) if roi_path else False

    # ── 처리 대상 쌍 목록 구성 ────────────────────────────────────────────────
    if lq_is_dir and ref_is_dir:
        # 폴더 모드: 동일 파일명으로 매칭
        lq_files = collect_image_files(lq_path)
        if not lq_files:
            raise FileNotFoundError(f"No image files found in lq_path: {lq_path}")

        image_pairs = []
        skipped = []
        roi_missing = []
        for fname in lq_files:
            ref_file = os.path.join(ref_path, fname)
            if not os.path.exists(ref_file):
                skipped.append(fname)
                continue
            # roi_path 가 폴더면 스템(확장자 제외 파일명)으로 매칭.
            # lq 가 .jpeg 이고 roi 가 .png 여도 스템이 같으면 매칭됨.
            if roi_is_dir:
                stem     = os.path.splitext(fname)[0]
                roi_file = find_roi_by_stem(roi_path, stem)
                if roi_file is None:
                    roi_missing.append(fname)
            else:
                roi_file = roi_path   # 파일 지정이거나 None
            image_pairs.append((
                os.path.join(lq_path, fname),
                ref_file,
                os.path.join(output_path, fname),
                roi_file,
            ))

        if skipped:
            print(f"[WARNING] {len(skipped)} file(s) skipped (no matching ref): {skipped}")
        if roi_missing:
            print(f"[WARNING] {len(roi_missing)} file(s) have no matching ROI mask "
                  f"(will use full-image ROI): {roi_missing}")
        if not image_pairs:
            raise FileNotFoundError(
                "No matching (lq, ref) pairs found. "
                "Make sure lq and ref folders contain images with identical filenames."
            )

        os.makedirs(output_path, exist_ok=True)
        _out_dir_for_cfg = output_path
        print(f">>> Folder mode: {len(image_pairs)} image pair(s) found.")
        print(f"    lq_path    : {lq_path}")
        print(f"    ref_path   : {ref_path}")
        print(f"    roi_path   : {roi_path or '(none)'}")
        print(f"    output_path: {output_path}")

    elif not lq_is_dir and not ref_is_dir:
        # 단일 파일 모드
        # output_path 가 디렉토리면 파일명 자동 결정
        if os.path.isdir(output_path):
            fname = os.path.basename(lq_path)
            out_file = os.path.join(output_path, fname)
            _out_dir_for_cfg = output_path
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            out_file = output_path
            _out_dir_for_cfg = out_dir or "."

        image_pairs = [(lq_path, ref_path, out_file, roi_path)]
        print(f">>> Single-image mode: {os.path.basename(lq_path)}")

    else:
        raise ValueError(
            "lq_path and ref_path must both be files or both be directories. "
            f"Got lq_is_dir={lq_is_dir}, ref_is_dir={ref_is_dir}"
        )

    # ── 실험 기록: effective config (YAML + CLI override 반영) 를 output 디렉터리에 저장 ──
    _cfg_src  = args.get("config_path", None)
    _cfg_name = os.path.basename(_cfg_src) if _cfg_src else "run_config.yaml"
    _cfg_save = os.path.join(_out_dir_for_cfg, _cfg_name)
    OmegaConf.save(args, _cfg_save)
    print(f">>> Config saved : {_cfg_save}")

    # ── 모델 로드 (전체 처리에 걸쳐 한 번만) ─────────────────────────────────
    print(">>> Loading models...")
    net_sr, net_ref, net_de, model_vlm, model_vlm_deg, ref_writer, ref_reader = \
        load_models(args, device, weight_dtype)
    print(">>> Models loaded.")

    # ── 이미지 쌍 순차 처리 ──────────────────────────────────────────────────
    n_total = len(image_pairs)
    for idx, (lq_f, ref_f, out_f, roi_f) in enumerate(image_pairs, 1):
        print(f"\n[{idx}/{n_total}] {os.path.basename(lq_f)}")
        _infer_single_image(
            lq_f, ref_f, out_f,
            net_sr, net_ref, net_de, model_vlm, model_vlm_deg,
            ref_writer, ref_reader,
            args, device, weight_dtype, tile_size, overlap, batch_sz, scale,
            visualize=visualize, roi_path=roi_f, vis_output_path=vis_out,
        )

    print(f"\n>>> All done. {n_total} image(s) processed.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_parser = argparse.ArgumentParser(add_help=False)
    demo_parser.add_argument("--config",          type=str, default="./configs/demo_tiled_config.yaml")
    demo_parser.add_argument("--lq_path",         type=str, default=None,
                             help="LQ image file or folder. Can also be set in YAML.")
    demo_parser.add_argument("--ref_path",        type=str, default=None,
                             help="Ref image file or folder (filenames must match lq). "
                                  "Can also be set in YAML.")
    demo_parser.add_argument("--output_path",     type=str, default=None,
                             help="Output file path (single mode) or folder (batch mode). "
                                  "Can also be set in YAML.")
    demo_parser.add_argument("--tile_size",       type=int, default=None,
                             help="Tile height=width in pixels (default: from YAML or 512).")
    demo_parser.add_argument("--tile_overlap",    type=int, default=None,
                             help="Overlap between adjacent tiles in pixels.")
    demo_parser.add_argument("--tile_batch_size", type=int, default=None,
                             help="Number of tiles processed per GPU batch.")
    demo_parser.add_argument("--scale",           type=int,  default=None,
                             help="SR upscale factor (default: from YAML or 4).")
    # [Fix 3] default=None: CLI 미지정 시 None → cli_overrides 에서 제외 → YAML 값 유지
    demo_parser.add_argument("--visualize",       action="store_true", default=None,
                             help="AICG Trust/Verify/Combined 히트맵 시각화 활성화.")
    demo_parser.add_argument("--roi_path",        type=str,  default=None,
                             help="ROI 마스크 이미지 경로 (grayscale PNG, LQ 해상도).")
    demo_parser.add_argument("--vis_output_path", type=str,  default=None,
                             help="시각화 PNG 저장 경로 (미지정 시 output_path 옆에 자동 생성).")

    # ── AICG Steering / Face Preprocessing flags ────────────────────────────
    demo_parser.add_argument("--enable_steering",     action="store_true", default=None,
                             help="AICGSteerer (Trust logit / Verify gate forcing) 활성화.")
    demo_parser.add_argument("--enable_face_preproc", action="store_true", default=None,
                             help="Face Soft mask 영역 LR Blur+Monochrome noise 전처리 활성화.")
    demo_parser.add_argument("--aicg_scale",          type=float, default=None,
                             help="전체 강화 배율 (trust/verify 양쪽에 곱함, default 1.0).")
    demo_parser.add_argument("--aicg_trust_scale",    type=float, default=None,
                             help="Trust(logit) 개별 배율.")
    demo_parser.add_argument("--aicg_verify_scale",   type=float, default=None,
                             help="Verify(gate G) 개별 배율 (sigma 통과 후 clamp).")
    demo_parser.add_argument("--aicg_force_verify",   action="store_true", default=None,
                             help="face 영역 gate 를 무조건 1.0 으로 강제 (Hard Override).")
    demo_parser.add_argument("--face_sigma_ratio",    type=float, default=None,
                             help="원본 face box 픽셀당 sigma 비율 (default 0.02).")
    demo_parser.add_argument("--face_noise_std",      type=float, default=None,
                             help="Monochrome 가우시안 노이즈 std ([-1,1] 공간, default 0.08).")
    demo_parser.add_argument("--face_blend_ratio",    type=float, default=None,
                             help="degrad_blend_ratio: 0=원본, 1=완전 degraded (default 0.6).")
    demo_parser.add_argument("--save_degraded_lr",    action="store_true", default=None,
                             help="blur/noise 가 적용된 LR 이미지를 output_path 옆에 저장.")
    demo_parser.add_argument("--degraded_lr_suffix",  type=str, default=None,
                             help="Degraded LR 파일명 접미사 (default '_degrad_lr').")

    demo_args, unknown = demo_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown
    args = parse_args_paired_testing()

    base_cfg = OmegaConf.create(vars(args))
    if os.path.exists(demo_args.config):
        yaml_cfg = OmegaConf.load(demo_args.config)
        base_cfg = OmegaConf.merge(base_cfg, yaml_cfg)
        print(f">>> Loaded YAML config from {demo_args.config}")

    # CLI 인수 중 None이 아닌 것만 덮어씀 (YAML 값 우선, CLI가 최우선)
    cli_overrides = {k: v for k, v in vars(demo_args).items()
                     if v is not None and k != 'config'}
    final_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(cli_overrides))

    # 필수 경로 확인
    for key in ('lq_path', 'ref_path', 'output_path'):
        if not final_cfg.get(key):
            raise ValueError(
                f"'{key}' is required. Set it via --{key} or in the YAML config."
            )

    # 실험 기록용: run_demo_tiled 내부에서 output 디렉터리에 config 사본 저장
    final_cfg.config_path = demo_args.config

    run_demo_tiled(final_cfg.lq_path, final_cfg.ref_path, final_cfg.output_path, final_cfg)
