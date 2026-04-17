import torch
import os
import argparse
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from main_code.model.gen_model import GenModel
from main_code.model.ref_model import RefModel
from main_code.model.de_net import DEResNet
from main_code.model.anymate_anyone.reference_attention import ReferenceNetAttention

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


def run_demo_tiled(lq_path, ref_path, output_path, args):
    accelerator  = Accelerator(mixed_precision=args.mixed_precision)
    device       = accelerator.device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    tile_size = int(args.get("tile_size",       512))
    overlap   = int(args.get("tile_overlap",     64))
    batch_sz  = int(args.get("tile_batch_size",   4))

    # ── Load models ──────────────────────────────────────────────────────────
    net_sr, net_ref, net_de, model_vlm, model_vlm_deg, ref_writer, ref_reader = \
        load_models(args, device, weight_dtype)

    # ── Preprocess images ────────────────────────────────────────────────────
    lq_img  = Image.open(lq_path).convert("RGB")
    ref_img = Image.open(ref_path).convert("RGB")

    # Align dimensions to multiples of 8 (VAE requirement)
    lq_img  = lq_img.resize( (lq_img.size[0]  // 8 * 8,  lq_img.size[1]  // 8 * 8))
    ref_img = ref_img.resize((ref_img.size[0] // 8 * 8, ref_img.size[1] // 8 * 8))
    orig_w, orig_h = lq_img.size

    to_tensor = transforms.ToTensor()
    x_lq  = to_tensor(lq_img)    # [3, H,  W ]
    x_ref = to_tensor(ref_img)   # [3, Hr, Wr]
    lq_h,  lq_w  = x_lq.shape[1],  x_lq.shape[2]
    ref_h, ref_w = x_ref.shape[1], x_ref.shape[2]

    # ── Global semantic prompts (computed once from the full images) ──────────
    with torch.no_grad():
        prompt_ref = inference(
            ram_transforms(x_ref.unsqueeze(0)).to(device, dtype=torch.float16), model_vlm)
        prompt_src = inference(
            ram_transforms(x_lq.unsqueeze(0)).to(device,  dtype=torch.float16), model_vlm_deg)
    print(f">>> Prompt (ref): {prompt_ref}")
    print(f">>> Prompt (src): {prompt_src}")

    # ── Single-tile fast-path (image already fits in one tile) ───────────────
    if lq_h <= tile_size and lq_w <= tile_size:
        print(">>> Image fits in one tile – running single inference.")
        x_src_t = x_lq.unsqueeze(0).to(device, dtype=weight_dtype)
        x_ref_t = x_ref.unsqueeze(0).to(device, dtype=weight_dtype)
        preds = infer_batch(net_sr, net_ref, net_de, ref_writer, ref_reader,
                            x_src_t, x_ref_t, prompt_src, prompt_ref, weight_dtype)
        pred_img = transforms.ToPILImage()((preds[0] * 0.5 + 0.5).clamp(0, 1).cpu())
        if args.get('align_method', 'wavelet') == 'wavelet':
            pred_img = wavelet_color_fix(pred_img, lq_img)
        pred_img.resize((orig_w, orig_h), Image.BICUBIC).save(output_path)
        print(f"✅ Result saved to: {output_path}")
        return

    # ── Build tile grid ───────────────────────────────────────────────────────
    ys = tile_start_coords(lq_h, tile_size, overlap)
    xs = tile_start_coords(lq_w, tile_size, overlap)
    all_tiles = [(y, x) for y in ys for x in xs]
    n_tiles    = len(all_tiles)
    print(f">>> Grid: {lq_h}×{lq_w} → {len(ys)}×{len(xs)} = {n_tiles} tiles "
          f"(tile_size={tile_size}, overlap={overlap}, batch={batch_sz})")

    # Output accumulators (float32 for precision)
    pred_acc   = torch.zeros(3, lq_h, lq_w, dtype=torch.float32, device=device)
    weight_acc = torch.zeros(1, lq_h, lq_w, dtype=torch.float32, device=device)

    # ── Tile inference loop ───────────────────────────────────────────────────
    for batch_start in range(0, n_tiles, batch_sz):
        batch_pos = all_tiles[batch_start : batch_start + batch_sz]
        B = len(batch_pos)
        print(f"    tiles {batch_start + 1}–{batch_start + B} / {n_tiles}")

        lq_tiles  = []
        ref_tiles = []
        for (ty, tx) in batch_pos:
            # LQ tile (always tile_size × tile_size due to grid construction)
            lq_tiles.append(x_lq[:, ty:ty + tile_size, tx:tx + tile_size])

            # Ref tile: extract a proportionally corresponding region, then
            # resize to tile_size so the model always sees the expected input shape.
            ry  = int(ty * ref_h / lq_h)
            rx  = int(tx * ref_w / lq_w)
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

        x_src_b = torch.stack(lq_tiles).to(device, dtype=weight_dtype)   # [B, 3, ts, ts]
        x_ref_b = torch.stack(ref_tiles).to(device, dtype=weight_dtype)

        # Repeat global prompts for each item in the batch
        prompts_ref_b = [prompt_ref] * B
        prompts_src_b = [prompt_src] * B

        predictions = infer_batch(
            net_sr, net_ref, net_de, ref_writer, ref_reader,
            x_src_b, x_ref_b, prompts_src_b, prompts_ref_b, weight_dtype,
        )

        # Weighted accumulation
        for k, (ty, tx) in enumerate(batch_pos):
            w    = tile_weight_map(tile_size, overlap, ty, tx, lq_h, lq_w).to(device)
            pred = (predictions[k] * 0.5 + 0.5).clamp(0, 1).float()   # [3, ts, ts]
            pred_acc[:,   ty:ty + tile_size, tx:tx + tile_size] += pred * w
            weight_acc[:, ty:ty + tile_size, tx:tx + tile_size] += w

    # ── Merge tiles ───────────────────────────────────────────────────────────
    result     = (pred_acc / weight_acc.clamp(min=1e-6)).clamp(0, 1)
    result_img = transforms.ToPILImage()(result.cpu())
    if args.get('align_method', 'wavelet') == 'wavelet':
        result_img = wavelet_color_fix(result_img, lq_img)
    result_img.resize((orig_w, orig_h), Image.BICUBIC).save(output_path)
    print(f"✅ Result saved to: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_parser = argparse.ArgumentParser(add_help=False)
    demo_parser.add_argument("--config",          type=str, default="./configs/demo_tiled_config.yaml")
    demo_parser.add_argument("--lq_path",         type=str, required=True)
    demo_parser.add_argument("--ref_path",        type=str, required=True)
    demo_parser.add_argument("--output_path",     type=str, default="./assets/pic/result_tiled.png")
    demo_parser.add_argument("--tile_size",       type=int, default=512,
                             help="Tile height=width in pixels. Should match the model's "
                                  "native resolution (default 512).")
    demo_parser.add_argument("--tile_overlap",    type=int, default=64,
                             help="Overlap between adjacent tiles in pixels. "
                                  "Larger values reduce seam artifacts but increase compute.")
    demo_parser.add_argument("--tile_batch_size", type=int, default=4,
                             help="Number of tiles processed per GPU batch.")

    demo_args, unknown = demo_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown
    args = parse_args_paired_testing()

    base_cfg = OmegaConf.create(vars(args))
    if os.path.exists(demo_args.config):
        yaml_cfg = OmegaConf.load(demo_args.config)
        base_cfg = OmegaConf.merge(base_cfg, yaml_cfg)
        print(f">>> Loaded YAML config from {demo_args.config}")

    final_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(vars(demo_args)))
    run_demo_tiled(final_cfg.lq_path, final_cfg.ref_path, final_cfg.output_path, final_cfg)
