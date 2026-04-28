"""
face_preproc.py – Tiling-aware face-region preprocessing (pixel space)
=======================================================================

VAE 인코더 통과 전 픽셀 공간에서 동작하는 전처리 유틸리티 모듈.

핵심 동작:
  1) Dynamic sigma   : 원본 해상도 face box 크기(max H,W)에 비례한 Gaussian sigma
                       (현재 degrade_lr_tile 내부에서는 미사용; soft_mask 생성 용도로만 호출됨)
  2) Mask softening  : Hard box mask → Gaussian blur → Soft mask
  3) LR degradation  : Soft mask 영역에 한해 Pixelization (block downsample → bilinear up)
                       + Monochrome 가우시안 노이즈
  4) Domain-shift 방어: blend_ratio 로 원본과 보간 + clamp(-1, 1) 로 VAE OOD 방지

Pixelization (replaces previous Gaussian blur path):
  얼굴 영역의 잘못된 ID 단서를 더 강하게 지우기 위해 Mosaic 형태로 다운/업샘플.
  block_size=8 기본값.  area downsample → bilinear upsample.

Monochrome noise:
  RGB 독립 노이즈는 chroma/색잡음을 만들어 ID 손상을 가속화함.
  단일 채널 [B,1,H,W] 노이즈를 3채널로 broadcast 하여 휘도 노이즈만 주입.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic blur sigma
# ─────────────────────────────────────────────────────────────────────────────

def compute_dynamic_sigma(
    face_box_max_px: float,
    sigma_ratio: float = 0.02,
    sigma_min:   float = 1.5,
    sigma_max:   float = 8.0,
) -> float:
    """원본 해상도 face box 크기에 비례한 Gaussian sigma.

    sigma = clamp(face_box_max_px * sigma_ratio, [sigma_min, sigma_max])

    Args:
        face_box_max_px : max(face_h, face_w) in pixels at original resolution
        sigma_ratio     : face box 픽셀 1px당 적용할 sigma 비율 (default 0.02)
                           예) 256px face → sigma ≈ 5.12
        sigma_min/max   : sigma 하한/상한 (안전 범위)
    """
    sigma = float(face_box_max_px) * float(sigma_ratio)
    return float(max(sigma_min, min(sigma_max, sigma)))


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian blur (separable, torch native)
# ─────────────────────────────────────────────────────────────────────────────

_kernel_cache: dict = {}   # keyed by (sigma, kernel_size, device_str, dtype_str)


def _gaussian_kernel_1d(
    sigma:       float,
    kernel_size: Optional[int] = None,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    if kernel_size is None:
        kernel_size = max(3, int(2 * round(3.0 * sigma) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1

    cache_key = (float(sigma), kernel_size, str(device), str(dtype))
    cached = _kernel_cache.get(cache_key)
    if cached is not None:
        return cached

    half = (kernel_size - 1) // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    _kernel_cache[cache_key] = k
    return k


def gaussian_blur_2d(
    x:           torch.Tensor,           # [B, C, H, W] / [C, H, W] / [H, W]
    sigma:       float,
    kernel_size: Optional[int] = None,
) -> torch.Tensor:
    """Separable 2D Gaussian blur (reflect padding). 입력 shape 그대로 반환."""
    if sigma <= 0:
        return x

    orig_ndim = x.ndim
    if orig_ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        x = x.unsqueeze(0)

    B, C, H, W = x.shape
    k = _gaussian_kernel_1d(sigma, kernel_size, device=x.device, dtype=x.dtype)
    K = k.shape[0]
    pad = K // 2

    kx = k.view(1, 1, 1, K).expand(C, 1, 1, K).contiguous()
    ky = k.view(1, 1, K, 1).expand(C, 1, K, 1).contiguous()

    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    x = F.conv2d(x, kx, groups=C)
    x = F.conv2d(x, ky, groups=C)

    if orig_ndim == 2:
        return x.squeeze(0).squeeze(0)
    if orig_ndim == 3:
        return x.squeeze(0)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Pixelization (mosaic) – area downsample then bilinear upsample
# ─────────────────────────────────────────────────────────────────────────────

def pixelize_2d(
    x:          torch.Tensor,   # [B, C, H, W] / [C, H, W] / [H, W]
    block_size: int,
) -> torch.Tensor:
    """Mosaic-style pixelization.

    H,W → (H/block, W/block) area pool → (H, W) bilinear upsample.
    block_size <= 1 이면 입력 그대로 반환 (no-op).
    입력 shape 그대로 반환.
    """
    if block_size is None or block_size <= 1:
        return x

    orig_ndim = x.ndim
    if orig_ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        x = x.unsqueeze(0)

    B, C, H, W = x.shape
    Hd = max(1, H // block_size)
    Wd = max(1, W // block_size)
    x_d = F.interpolate(x, size=(Hd, Wd), mode='area')
    x_u = F.interpolate(x_d, size=(H, W), mode='bilinear', align_corners=False)

    if orig_ndim == 2:
        return x_u.squeeze(0).squeeze(0)
    if orig_ndim == 3:
        return x_u.squeeze(0)
    return x_u


# ─────────────────────────────────────────────────────────────────────────────
# Mask softening
# ─────────────────────────────────────────────────────────────────────────────

def soften_mask(
    box_mask: torch.Tensor,   # [H, W] / [B, 1, H, W]  (0/1 hard mask 권장)
    sigma:    float,
) -> torch.Tensor:
    """Hard box mask 에 Gaussian blur 적용 → Soft mask (값 [0, 1])."""
    if sigma <= 0:
        return box_mask.clamp(0, 1)
    blurred = gaussian_blur_2d(box_mask.float(), sigma)
    return blurred.clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Monochrome Gaussian noise
# ─────────────────────────────────────────────────────────────────────────────

def monochrome_gaussian_noise(
    image: torch.Tensor,   # [B, C, H, W] / [C, H, W]
    std:   float,
) -> torch.Tensor:
    """단일 채널 가우시안 노이즈를 모든 채널로 broadcast (chroma 노이즈 방지).

    - RGB 독립 노이즈와 달리 색잡음을 만들지 않고 휘도 노이즈만 주입.
    - 결과 shape 은 입력과 동일.
    """
    if std <= 0:
        return torch.zeros_like(image)

    if image.ndim == 3:
        C, H, W = image.shape
        n = torch.randn(1, H, W, device=image.device, dtype=image.dtype) * std
        return n.expand(C, H, W).contiguous()
    elif image.ndim == 4:
        B, C, H, W = image.shape
        n = torch.randn(B, 1, H, W, device=image.device, dtype=image.dtype) * std
        return n.expand(B, C, H, W).contiguous()
    else:
        raise ValueError(f"monochrome_gaussian_noise: ndim={image.ndim} 미지원")


# ─────────────────────────────────────────────────────────────────────────────
# Full degradation pipeline (per-tile)
# ─────────────────────────────────────────────────────────────────────────────

def degrade_lr_tile(
    lq_tile_01:        torch.Tensor,   # [3, H, W] in [0, 1]
    soft_mask:         torch.Tensor,   # [H, W]    in [0, 1]
    sigma:             float,
    noise_std:         float,
    degrad_blend_ratio: float,
    clamp_range:       Tuple[float, float] = (-1.0, 1.0),
    pixel_block_size:  int = 8,
) -> torch.Tensor:
    """Soft mask 영역에 한해 LR 이미지에 Pixelization + Monochrome 노이즈 주입.

    수식 (VAE input 공간 [-1, 1]):
        x         = lq_tile_01 * 2 - 1
        pixelized = AreaDownsample(x, /block) → BilinearUpsample(., H,W)
        degraded  = pixelized + monochrome_noise(std=noise_std)
        mixed     = blend * degraded + (1 - blend) * x          (도메인 시프트 방어)
        result    = soft_mask * mixed + (1 - soft_mask) * x     (얼굴 영역만 적용)
        result    = clamp(result, -1, 1)                        (VAE OOD 방지)
        return      (result + 1) / 2                            (다시 [0, 1])

    Args:
        lq_tile_01         : LR 타일 [3, H, W], [0, 1] 범위
        soft_mask          : 같은 H,W 의 soft mask, [0, 1] 범위
        sigma              : (Deprecated) Gaussian blur sigma — 현재 사용되지 않음.
                              과거 시그니처 호환을 위해 남겨둠. soft_mask 생성용 sigma 와는 무관.
        noise_std          : monochrome 노이즈 std (in [-1,1] 공간)
        degrad_blend_ratio : 0 = no degradation, 1 = full degradation in mask
        clamp_range        : VAE OOD 방지용 clamp 범위 (default [-1, 1])
        pixel_block_size   : Pixelization 블록 크기 (default 8). <=1 이면 pixelization 비활성.

    Returns:
        degraded LR tile [3, H, W] in [0, 1]
    """
    assert lq_tile_01.ndim == 3, f"lq_tile_01 ndim={lq_tile_01.ndim} (expected 3)"
    assert soft_mask.ndim == 2,   f"soft_mask  ndim={soft_mask.ndim} (expected 2)"

    device = lq_tile_01.device
    dtype  = lq_tile_01.dtype

    x = lq_tile_01 * 2.0 - 1.0                                   # → [-1, 1]
    pixelized = pixelize_2d(x, pixel_block_size)
    noise     = monochrome_gaussian_noise(pixelized, noise_std)
    degraded  = pixelized + noise

    m = soft_mask.to(device=device, dtype=dtype).unsqueeze(0)    # [1, H, W]

    blend = float(degrad_blend_ratio)
    mixed = blend * degraded + (1.0 - blend) * x

    result = m * mixed + (1.0 - m) * x
    # 핵심 안전장치: VAE OOD 방지
    result = torch.clamp(result, min=clamp_range[0], max=clamp_range[1])

    return (result + 1.0) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def bbox_max_size_from_mask(mask) -> int:
    """0이 아닌 영역의 bounding box 에서 max(H, W) 를 픽셀 단위로 반환.

    mask : np.ndarray 또는 torch.Tensor, 임의 dtype, 임의 ndim (≥ 2)
    return: 픽셀 길이 (mask 가 모두 0이면 0)
    """
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)
    while m.ndim > 2:
        m = m.squeeze()
        if m.ndim == mask.ndim:
            break
    if m.ndim > 2:
        # squeeze 가 안 되는 경우 first-axis projection
        m = m.reshape(-1, m.shape[-1])
    if m.dtype.kind in ('f', 'd'):
        nz = np.argwhere(m > 0.5)
    else:
        nz = np.argwhere(m > 0)
    if nz.size == 0:
        return 0
    y0, x0 = nz.min(0)
    y1, x1 = nz.max(0)
    return int(max(y1 - y0 + 1, x1 - x0 + 1))
