import torch
import os
import argparse
import sys
import math
from typing import Optional
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from main_code.model.gen_model import GenModel
from main_code.model.ref_model import RefModel
from main_code.model.de_net import DEResNet
from main_code.model.anymate_anyone.reference_attention import ReferenceNetAttention
from visualize_aicg import AICGVisualizer
from aicg_steering import AICGSteerer

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


def find_file_by_stem(folder, stem):
    """Find first image in folder whose stem matches (extension-agnostic)."""
    for fname in os.listdir(folder):
        s, ext = os.path.splitext(fname)
        if s == stem and ext.lower() in SUPPORTED_EXTENSIONS:
            return os.path.join(folder, fname)
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
    """Run one batched SR forward pass (Per-tile ref crop mode).

    각 tile-batch 마다 net_ref 를 새 ref crop 으로 재실행 → ref_reader.update → net_sr → clear.
    Returns raw model output [B,3,H,W] in [-1,1].
    """
    with torch.no_grad():
        deg_scores  = net_de(x_src_b)
        net_ref(x_ref_b * 2.0 - 1.0, prompt=prompts_ref)
        ref_reader.update(ref_writer, dtype=weight_dtype)
        predictions = net_sr(x_src_b * 2.0 - 1.0, deg_scores, prompt=prompts_src)
        ref_reader.clear()
        ref_writer.clear()
    return predictions


def infer_global_ref(net_ref, ref_writer, ref_reader, x_ref_g, prompt_ref, weight_dtype):
    """[Full Ref Attention] net_ref 를 글로벌 ref 1장에 대해 1회만 실행, ref_reader 채움."""
    with torch.no_grad():
        net_ref(x_ref_g * 2.0 - 1.0, prompt=[prompt_ref])
        ref_reader.update(ref_writer, dtype=weight_dtype)


def infer_sr_only(net_sr, net_de, x_src_b, prompts_src, weight_dtype):
    """[Full Ref Attention] SR forward 만 실행 (ref_reader 가 이미 채워져 있다고 가정).

    Returns raw model output [B,3,H,W] in [-1,1].
    """
    with torch.no_grad():
        deg_scores  = net_de(x_src_b)
        predictions = net_sr(x_src_b * 2.0 - 1.0, deg_scores, prompt=prompts_src)
    return predictions


def _infer_single_image(lq_path, ref_path, output_path,
                        net_sr, net_ref, net_de, model_vlm, model_vlm_deg,
                        ref_writer, ref_reader, args, device, weight_dtype,
                        tile_size, overlap, batch_sz, scale,
                        visualize=False, vis_output_path=None,
                        fmt=None):
    """Process a single (lq, ref) image pair and save the result."""

    # ── Load & align original images ─────────────────────────────────────────
    lq_img  = ImageOps.exif_transpose(Image.open(lq_path).convert("RGB"))
    ref_img = ImageOps.exif_transpose(Image.open(ref_path).convert("RGB"))

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

    # ── Bicubic upscale LQ by `scale` ────────────────────────────────────────
    sr_h = orig_h * scale // 8 * 8
    sr_w = orig_w * scale // 8 * 8
    x_lq = F.interpolate(
        x_lq_orig.unsqueeze(0),
        size=(sr_h, sr_w),
        mode='bicubic', align_corners=False,
    ).squeeze(0).clamp(0, 1)          # [3, H*scale, W*scale]

    lq_h, lq_w = x_lq.shape[1], x_lq.shape[2]

    # ── Optional Gaussian blur on 4x canvas (FMT uses x_lq_orig, unaffected) ─
    lq_blur_sigma = float(args.get("lq_blur_sigma", 0.0))
    if lq_blur_sigma > 0.0:
        _ks = 2 * int(math.ceil(3.0 * lq_blur_sigma)) + 1
        x_lq = TF.gaussian_blur(x_lq, kernel_size=_ks, sigma=lq_blur_sigma).clamp(0, 1)
        print(f"  LQ blur: sigma={lq_blur_sigma:.2f}, kernel={_ks}x{_ks}")

    lq_bicubic_img = transforms.ToPILImage()(x_lq.cpu())

    print(f"  LQ: {orig_w}x{orig_h}  ->  SR canvas: {lq_w}x{lq_h}  (x{scale})")

    # ── Global semantic prompts ───────────────────────────────────────────────
    with torch.no_grad():
        prompt_ref = inference(
            ram_transforms(x_ref.unsqueeze(0)).to(device, dtype=torch.float16),
            model_vlm)[0]
        prompt_src = inference(
            ram_transforms(x_lq_orig.unsqueeze(0)).to(device, dtype=torch.float16),
            model_vlm_deg)[0]
    print(f"  Prompt (ref): {prompt_ref}")
    print(f"  Prompt (src): {prompt_src}")

    # ── AICG steering 설정 ────────────────────────────────────────────────────
    enable_steering   = bool(args.get("enable_steering",    False))
    aicg_scale        = float(args.get("aicg_scale",        1.0))
    aicg_trust_scale  = float(args.get("aicg_trust_scale",  1.0))
    aicg_verify_scale = float(args.get("aicg_verify_scale", 1.0))
    aicg_force_verify = bool(args.get("aicg_force_verify",  False))

    enable_full_ref = bool(args.get("enable_full_ref",   False))
    full_ref_size   = int(args.get("full_ref_size",      1024))

    # AICGSteerer 초기화
    steerer: Optional[AICGSteerer] = None
    if enable_steering:
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

    # ── Global ref preparation (Full Ref Attention 모드) ─────────────────────
    x_ref_global: Optional[torch.Tensor] = None
    if enable_full_ref:
        x_ref_global = F.interpolate(
            x_ref.unsqueeze(0),
            size=(full_ref_size, full_ref_size),
            mode='bicubic', align_corners=False,
        ).squeeze(0).clamp(0, 1)
        ref_h_eff, ref_w_eff = full_ref_size, full_ref_size
        ref_img_for_viz = transforms.ToPILImage()(x_ref_global.cpu())
        print(f"  Full Ref Attention ON: ref resized {ref_h}x{ref_w} -> "
              f"{full_ref_size}x{full_ref_size}, shared by all LR tiles")
    else:
        ref_h_eff, ref_w_eff = ref_h, ref_w
        ref_img_for_viz = ref_img

    # ── Single-tile fast-path (upscaled LQ fits in one tile) ─────────────────
    if lq_h <= tile_size and lq_w <= tile_size:
        print("  SR canvas fits in one tile – running single inference.")

        x_src_t = x_lq.unsqueeze(0).to(device, dtype=weight_dtype)

        # 단일 타일에서 FMT 미사용 시 전역 steering 적용;
        # FMT 사용 시에는 타일 매칭 없이 단일 타일이므로 steering 없음.
        _steer_single = (steerer is not None and fmt is None)

        if enable_full_ref:
            x_ref_t = x_ref_global.unsqueeze(0).to(device, dtype=weight_dtype)
            infer_global_ref(net_ref, ref_writer, ref_reader, x_ref_t, prompt_ref, weight_dtype)
            def _run_infer():
                return infer_sr_only(net_sr, net_de, x_src_t, [prompt_src], weight_dtype)
        else:
            x_ref_t = x_ref.unsqueeze(0).to(device, dtype=weight_dtype)
            def _run_infer():
                return infer_batch(
                    net_sr, net_ref, net_de, ref_writer, ref_reader,
                    x_src_t, x_ref_t, [prompt_src], [prompt_ref], weight_dtype,
                )

        if visualize:
            _vis_output = _resolve_vis_path(vis_output_path, output_path)
            if steerer is not None:
                steerer.capture_for_viz = True
            viz = AICGVisualizer(
                net_sr.unet,
                ref_h=ref_h_eff, ref_w=ref_w_eff,
                lq_h=lq_h, lq_w=lq_w,
                tile_size=tile_size,
                fusion_blocks=args.get("fusion_blocks", "full"),
                steerer=steerer,
            )
            with viz.capture_tile(0, 0, 0, 0, ref_h_eff, ref_w_eff, lq_h, lq_w):
                if _steer_single:
                    with steerer.apply_steering():
                        preds = _run_infer()
                else:
                    preds = _run_infer()
            viz.finalize(np.array(ref_img_for_viz), np.array(lq_bicubic_img), _vis_output)
        else:
            if _steer_single:
                with steerer.apply_steering():
                    preds = _run_infer()
            else:
                preds = _run_infer()

        if enable_full_ref:
            ref_writer.clear()
            ref_reader.clear()

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
    n_tiles   = len(all_tiles)

    # ── Feature matching preparation (once per image, before any batch) ──────
    if fmt is not None:
        fmt_ready = fmt.prepare(x_lq_orig, x_ref)
        if fmt_ready:
            if args.get("visualize_feature_matching", False):
                _stem = os.path.splitext(os.path.basename(output_path))[0]
                _fmt_vis_path = os.path.join(
                    os.path.dirname(os.path.abspath(output_path)),
                    _stem + "_fmt_vis.png",
                )
                fmt.save_match_visualization(
                    save_path=_fmt_vis_path,
                    lr=x_lq_orig,
                    ref=x_ref,
                    tile_size_4x=tile_size,
                    overlap_4x=overlap,
                    scale=scale,
                )
            fmt._mkpts_lr *= scale   # 원본 LR → 4x canvas 좌표계로 변환
        else:
            print("  [FMT] Feature matching failed – proportional fallback for all tiles.")
    print(f"  Grid: {lq_h}x{lq_w} -> {len(ys)}x{len(xs)} = {n_tiles} tiles "
          f"(tile={tile_size}, overlap={overlap}, batch={batch_sz})")

    # Output accumulators at the SR canvas size (float32 for precision)
    pred_acc   = torch.zeros(3, lq_h, lq_w, dtype=torch.float32, device=device)
    weight_acc = torch.zeros(1, lq_h, lq_w, dtype=torch.float32, device=device)

    # ── 시각화 캔버스 초기화 ─────────────────────────────────────────────────
    viz = None
    if visualize:
        _vis_output = _resolve_vis_path(vis_output_path, output_path)
        if steerer is not None:
            steerer.capture_for_viz = True
        viz = AICGVisualizer(
            net_sr.unet,
            ref_h=ref_h_eff, ref_w=ref_w_eff,
            lq_h=lq_h, lq_w=lq_w,
            tile_size=tile_size,
            fusion_blocks=args.get("fusion_blocks", "full"),
            steerer=steerer,
        )

    # ── Full Ref Attention 모드: ref 1회만 forward ────────────────────────────
    if enable_full_ref:
        x_ref_g_t = x_ref_global.unsqueeze(0).to(device, dtype=weight_dtype)
        infer_global_ref(net_ref, ref_writer, ref_reader, x_ref_g_t, prompt_ref, weight_dtype)

    # ── Split tiles: FMT-matched vs proportional ──────────────────────────────
    # FMT ON: matched tiles → steering ON; proportional → steering OFF.
    # FMT OFF: all tiles treated as 'prop' (steering controlled by enable_steering alone).
    _precomp_crops:  dict = {}   # (ty, tx) -> crop tensor
    _precomp_coords: dict = {}   # (ty, tx) -> (ry, rx, rth, rtw)
    fmt_matched_tiles: list = []
    proportional_tiles: list = []

    if fmt is not None and fmt._ready and not enable_full_ref:
        _all_crops, _all_coords, _all_matched = fmt.get_ref_crops_batch(
            all_tiles, tile_size, x_ref, tile_size, lq_h, lq_w,
        )
        for _i, (_ty, _tx) in enumerate(all_tiles):
            _precomp_crops[(_ty, _tx)]  = _all_crops[_i]
            _precomp_coords[(_ty, _tx)] = _all_coords[_i]
            if _all_matched[_i]:
                fmt_matched_tiles.append((_ty, _tx))
            else:
                proportional_tiles.append((_ty, _tx))
        print(f"  [FMT] {len(fmt_matched_tiles)} matched tiles (steering ON), "
              f"{len(proportional_tiles)} proportional tiles (steering OFF)")
    else:
        proportional_tiles = all_tiles

    def _make_batches(positions, label):
        return [(label, positions[i : i + batch_sz])
                for i in range(0, len(positions), batch_sz)]

    # FMT-matched batches first (get steering), then proportional (no steering)
    all_batches   = _make_batches(fmt_matched_tiles, 'fmt') + _make_batches(proportional_tiles, 'prop')
    total_batches = len(all_batches)

    # ── Tile inference loop ───────────────────────────────────────────────────
    for batch_idx, (tile_type, batch_pos) in enumerate(all_batches):
        B = len(batch_pos)
        print(f"    batch {batch_idx + 1}/{total_batches} [{tile_type}]  ({B} tiles)")

        lq_tiles    = []
        ref_tiles   = []
        tile_coords = []

        for i_tile, (ty, tx) in enumerate(batch_pos):
            lq_tile = x_lq[:, ty:ty + tile_size, tx:tx + tile_size]
            lq_tiles.append(lq_tile)

            if enable_full_ref:
                tile_coords.append((ty, tx, 0, 0, ref_h_eff, ref_w_eff))
            elif (ty, tx) in _precomp_crops:
                # Feature-matching ref crop (pre-computed)
                ref_tiles.append(_precomp_crops[(ty, tx)])
                ry, rx, rth, rtw = _precomp_coords[(ty, tx)]
                tile_coords.append((ty, tx, ry, rx, rth, rtw))
            else:
                # Proportional fallback crop
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
        prompts_ref_b = [prompt_ref] * B
        prompts_src_b = [prompt_src] * B

        if enable_full_ref:
            def _run_batch():
                return infer_sr_only(net_sr, net_de, x_src_b, prompts_src_b, weight_dtype)
        else:
            x_ref_b = torch.stack(ref_tiles).to(device, dtype=weight_dtype)
            def _run_batch():
                return infer_batch(
                    net_sr, net_ref, net_de, ref_writer, ref_reader,
                    x_src_b, x_ref_b, prompts_src_b, prompts_ref_b, weight_dtype,
                )

        # Steering: FMT-matched tiles (tile_type='fmt') or all tiles when FMT not used
        _apply_steer = steerer is not None and (tile_type == 'fmt' or fmt is None)

        if viz is not None and _apply_steer:
            with viz.capture_batch(tile_coords, lq_h, lq_w), steerer.apply_steering():
                predictions = _run_batch()
        elif viz is not None:
            with viz.capture_batch(tile_coords, lq_h, lq_w):
                predictions = _run_batch()
        elif _apply_steer:
            with steerer.apply_steering():
                predictions = _run_batch()
        else:
            predictions = _run_batch()

        for k, (ty, tx) in enumerate(batch_pos):
            w    = tile_weight_map(tile_size, overlap, ty, tx, lq_h, lq_w).to(device)
            pred = (predictions[k] * 0.5 + 0.5).clamp(0, 1).float()
            pred_acc[:,   ty:ty + tile_size, tx:tx + tile_size] += pred * w
            weight_acc[:, ty:ty + tile_size, tx:tx + tile_size] += w

    # ── Full Ref: 모든 tile 처리 후 글로벌 ref bank 1회 clear ────────────────
    if enable_full_ref:
        ref_writer.clear()
        ref_reader.clear()

    # ── Merge tiles & save ────────────────────────────────────────────────────
    result     = (pred_acc / weight_acc.clamp(min=1e-6)).clamp(0, 1)
    result_img = transforms.ToPILImage()(result.cpu())
    if args.get('align_method', 'wavelet') == 'wavelet':
        result_img = wavelet_color_fix(result_img, lq_bicubic_img)
    result_img.resize((orig_w * scale, orig_h * scale), Image.BICUBIC).save(output_path)
    print(f"  Saved: {output_path}  ({orig_w * scale}x{orig_h * scale})")

    # ── AICG 시각화 저장 ─────────────────────────────────────────────────────
    if viz is not None:
        viz.finalize(np.array(ref_img_for_viz), np.array(lq_bicubic_img), _vis_output)


def run_demo_tiled(lq_path, ref_path, output_path, args):
    """
    폴더 경로가 주어지면 lq_path / ref_path 안의 동일 파일명 이미지를 전부 처리.
    파일 경로가 주어지면 단일 이미지 처리 (기존 동작).
    """
    accelerator  = Accelerator(mixed_precision=args.mixed_precision)
    device       = accelerator.device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    tile_size  = int(args.get("tile_size",       512))
    overlap    = int(args.get("tile_overlap",     64))
    batch_sz   = int(args.get("tile_batch_size",   4))
    scale      = int(args.get("scale",             4))
    visualize  = bool(args.get("visualize",      False))
    vis_out    = args.get("vis_output_path",     None)

    lq_is_dir  = os.path.isdir(lq_path)
    ref_is_dir = os.path.isdir(ref_path)

    # ── 처리 대상 쌍 목록 구성 ────────────────────────────────────────────────
    if lq_is_dir and ref_is_dir:
        lq_files = collect_image_files(lq_path)
        if not lq_files:
            raise FileNotFoundError(f"No image files found in lq_path: {lq_path}")

        image_pairs = []
        skipped = []
        for fname in lq_files:
            stem     = os.path.splitext(fname)[0]
            ref_file = find_file_by_stem(ref_path, stem)
            if ref_file is None:
                skipped.append(fname)
                continue
            image_pairs.append((
                os.path.join(lq_path, fname),
                ref_file,
                os.path.join(output_path, fname),
            ))

        if skipped:
            print(f"[WARNING] {len(skipped)} file(s) skipped (no matching ref): {skipped}")
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
        print(f"    output_path: {output_path}")

    elif not lq_is_dir and not ref_is_dir:
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

        image_pairs = [(lq_path, ref_path, out_file)]
        print(f">>> Single-image mode: {os.path.basename(lq_path)}")

    else:
        raise ValueError(
            "lq_path and ref_path must both be files or both be directories. "
            f"Got lq_is_dir={lq_is_dir}, ref_is_dir={ref_is_dir}"
        )

    # ── 실험 기록: effective config 를 output 디렉터리에 저장 ──────────────────
    _cfg_src  = args.get("config_path", None)
    _cfg_name = os.path.basename(_cfg_src) if _cfg_src else "run_config.yaml"
    _cfg_save = os.path.join(_out_dir_for_cfg, _cfg_name)
    OmegaConf.save(args, _cfg_save)
    print(f">>> Config saved : {_cfg_save}")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    print(">>> Loading models...")
    net_sr, net_ref, net_de, model_vlm, model_vlm_deg, ref_writer, ref_reader = \
        load_models(args, device, weight_dtype)
    print(">>> Models loaded.")

    # ── FeatureMatchingTiler (XFeat) 로드 ────────────────────────────────────
    fmt = None
    if bool(args.get("use_feature_matching_tiling", False)):
        xfeat_path = str(args.get("xfeat_project_path", "") or "")
        if xfeat_path and os.path.isdir(xfeat_path):
            from my_utils.feature_matching_tiling import FeatureMatchingTiler
            fmt = FeatureMatchingTiler(
                xfeat_path,
                ransac_threshold=float(args.get("fmt_ransac_threshold", 3.5)),
                min_inliers=int(args.get("fmt_min_inliers", 4)),
                match_at_ref_resolution=bool(args.get("fmt_match_at_ref_resolution", False)),
                matcher=str(args.get("fmt_matcher", "xfeat")),
            )
            print(f">>> FeatureMatchingTiler: XFeat loaded from {xfeat_path}")
        else:
            print(f"[WARNING] use_feature_matching_tiling=true but "
                  f"xfeat_project_path not found: '{xfeat_path}'")

    # ── 이미지 쌍 순차 처리 ──────────────────────────────────────────────────
    n_total = len(image_pairs)
    for idx, (lq_f, ref_f, out_f) in enumerate(image_pairs, 1):
        print(f"\n[{idx}/{n_total}] {os.path.basename(lq_f)}")
        _infer_single_image(
            lq_f, ref_f, out_f,
            net_sr, net_ref, net_de, model_vlm, model_vlm_deg,
            ref_writer, ref_reader,
            args, device, weight_dtype, tile_size, overlap, batch_sz, scale,
            visualize=visualize, vis_output_path=vis_out,
            fmt=fmt,
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
    demo_parser.add_argument("--visualize",       action="store_true", default=None,
                             help="AICG Trust/Verify/Combined 히트맵 시각화 활성화.")
    demo_parser.add_argument("--vis_output_path", type=str,  default=None,
                             help="시각화 PNG 저장 경로 (미지정 시 output_path 옆에 자동 생성).")

    # ── LR Blur ──────────────────────────────────────────────────────────────
    demo_parser.add_argument("--lq_blur_sigma", type=float, default=None,
                             help="4x canvas LR 에 적용할 Gaussian blur sigma "
                                  "(default 0.0 = no blur). FMT 매칭은 blur 전 원본 LR 사용.")

    # ── AICG Steering ────────────────────────────────────────────────────────
    demo_parser.add_argument("--enable_steering",     action="store_true", default=None,
                             help="AICGSteerer (Trust logit / Verify gate forcing) 활성화. "
                                  "FMT ON 시: matched tile에만 적용. FMT OFF 시: 모든 tile에 적용.")
    demo_parser.add_argument("--aicg_scale",          type=float, default=None,
                             help="전체 강화 배율 (trust/verify 양쪽에 곱함, default 1.0).")
    demo_parser.add_argument("--aicg_trust_scale",    type=float, default=None,
                             help="Trust(logit) 개별 배율.")
    demo_parser.add_argument("--aicg_verify_scale",   type=float, default=None,
                             help="Verify(gate G) 개별 배율.")
    demo_parser.add_argument("--aicg_force_verify",   action="store_true", default=None,
                             help="gate 를 무조건 1.0 으로 강제 (Hard Override).")

    # ── Full Ref Attention ───────────────────────────────────────────────────
    demo_parser.add_argument("--enable_full_ref",     action="store_true", default=None,
                             help="ref 를 1회만 (full_ref_size, full_ref_size) 로 resize 후 "
                                  "모든 LR tile 이 같은 ref bank 를 공유.")
    demo_parser.add_argument("--full_ref_size",       type=int,   default=None,
                             help="Full Ref Attention 시 ref 의 H=W (default 1024).")

    # ── XFeat Feature Matching Tiling ────────────────────────────────────────
    demo_parser.add_argument("--use_feature_matching_tiling", action="store_true", default=None,
                             help="XFeat 특징점 매칭으로 ref 타일을 동적 선택. "
                                  "enable_steering=true 와 함께 사용하면 matched tile만 steering됨.")
    demo_parser.add_argument("--xfeat_project_path", type=str, default=None,
                             help="accelerated_features 폴더의 절대/상대 경로.")

    demo_args, unknown = demo_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown
    args = parse_args_paired_testing()

    base_cfg = OmegaConf.create(vars(args))
    if os.path.exists(demo_args.config):
        yaml_cfg = OmegaConf.load(demo_args.config)
        base_cfg = OmegaConf.merge(base_cfg, yaml_cfg)
        print(f">>> Loaded YAML config from {demo_args.config}")

    cli_overrides = {k: v for k, v in vars(demo_args).items()
                     if v is not None and k != 'config'}
    final_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(cli_overrides))

    for key in ('lq_path', 'ref_path', 'output_path'):
        if not final_cfg.get(key):
            raise ValueError(
                f"'{key}' is required. Set it via --{key} or in the YAML config."
            )

    final_cfg.config_path = demo_args.config

    run_demo_tiled(final_cfg.lq_path, final_cfg.ref_path, final_cfg.output_path, final_cfg)
