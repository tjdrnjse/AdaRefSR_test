"""
XFeat-based Dynamic Feature Matching Tiling for AdaRefSR.

Pipeline per image pair (call prepare() once):
  1. Scale Sync    : resize ref to lr resolution
  2. Feature Match : XFeat sparse matching (match_xfeat)
  3. RANSAC        : cv2.findHomography inlier filtering (USAC_MAGSAC)
  4. Re-project    : scale surviving ref keypoints back to original ref space

Per-tile crop — Strategy 1, Median-based Crop (call get_ref_crops_batch()):
  5. Filter        : find lr keypoints inside current lr tile boundary
  6. Median crop   : median(ref pts) → center of fixed-size BBox → clamp+pad
  7. Fallback      : proportional crop if no keypoints fall in tile

GPU batch optimisation: all fallback proportional crops that need resizing are
stacked into a single F.interpolate call instead of per-tile calls.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import List, Optional, Tuple


class FeatureMatchingTiler:
    """XFeat-based ref tile selector for AdaRefSR tiled inference."""

    def __init__(self, xfeat_project_path: str):
        """
        xfeat_project_path: absolute path to the accelerated_features folder.
        XFeat is dynamically imported and uses CUDA when available.
        """
        if xfeat_project_path not in sys.path:
            sys.path.insert(0, xfeat_project_path)
        from modules.xfeat import XFeat
        self.xfeat = XFeat()

        self._ready = False
        self._mkpts_lr: Optional[np.ndarray] = None   # (N,2) x,y in LR canvas space
        self._mkpts_ref: Optional[np.ndarray] = None  # (N,2) x,y in original ref space

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_uint8(t: torch.Tensor) -> np.ndarray:
        """[3,H,W] float [0,1] CPU tensor → [H,W,3] uint8 numpy array."""
        return (t.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _bbox_from_median(
        ref_pts: np.ndarray,
        ref_h: int, ref_w: int,
        size: int,
    ) -> Tuple[int, int, int, int]:
        """
        Compute (x0, y0, x1, y1) BBox of fixed `size` centered on the median
        of ref_pts, clamped so the box stays inside the image.
        """
        half = size // 2
        cx = int(np.median(ref_pts[:, 0]))
        cy = int(np.median(ref_pts[:, 1]))

        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + size, y0 + size

        # slide window within image boundary
        if x0 < 0:     x0, x1 = 0, size
        if y0 < 0:     y0, y1 = 0, size
        if x1 > ref_w: x0, x1 = max(0, ref_w - size), ref_w
        if y1 > ref_h: y0, y1 = max(0, ref_h - size), ref_h

        return x0, y0, x1, y1

    @staticmethod
    def _crop_or_pad(
        ref: torch.Tensor,
        x0: int, y0: int, x1: int, y1: int,
        size: int,
    ) -> torch.Tensor:
        """
        Crop [y0:y1, x0:x1] from ref [3,H,W].
        Adds reflect-padding when the bbox is partially out of bounds (ref smaller than size).
        Returns [3, size, size] float32 CPU tensor.
        """
        ref_h, ref_w = ref.shape[1], ref.shape[2]
        xc0 = max(0, x0);  yc0 = max(0, y0)
        xc1 = min(ref_w, x1);  yc1 = min(ref_h, y1)
        crop = ref[:, yc0:yc1, xc0:xc1].float()

        pad_l = xc0 - x0
        pad_t = yc0 - y0
        pad_r = max(0, (x1 - x0) - crop.shape[2] - pad_l)
        pad_b = max(0, (y1 - y0) - crop.shape[1] - pad_t)

        if pad_l > 0 or pad_r > 0 or pad_t > 0 or pad_b > 0:
            crop = F.pad(crop, (pad_l, pad_r, pad_t, pad_b), mode='reflect')

        # safety resize in degenerate cases (ref much smaller than size)
        if crop.shape[1] != size or crop.shape[2] != size:
            crop = F.interpolate(
                crop.unsqueeze(0), size=(size, size),
                mode='bicubic', align_corners=False,
            ).squeeze(0).clamp(0, 1)

        return crop

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare(self, lr: torch.Tensor, ref: torch.Tensor) -> bool:
        """
        Run the full matching pipeline once per (lr, ref) image pair.

        lr  : [3, H,  W ] in [0,1] — upscaled LQ canvas (CPU tensor)
        ref : [3, Hr, Wr] in [0,1] — original ref image  (CPU tensor)

        Returns True when >= 4 RANSAC inlier matches are found.
        Results are stored internally for subsequent get_ref_crops_batch() calls.
        """
        self._ready = False
        self._mkpts_lr = None
        self._mkpts_ref = None

        lr_h, lr_w   = lr.shape[1],  lr.shape[2]
        ref_h, ref_w = ref.shape[1], ref.shape[2]

        # 1. Scale Sync: resize ref to lr resolution
        ref_scaled = F.interpolate(
            ref.unsqueeze(0).float(), size=(lr_h, lr_w),
            mode='bicubic', align_corners=False,
        ).squeeze(0).clamp(0, 1)

        # 2. Feature Matching
        lr_np  = self._to_uint8(lr)
        ref_np = self._to_uint8(ref_scaled)
        mkpts_lr, mkpts_ref_s = self.xfeat.match_xfeat(lr_np, ref_np)

        if len(mkpts_lr) < 4:
            return False

        # 3. RANSAC outlier removal
        method = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)
        _, mask = cv2.findHomography(
            mkpts_lr.astype(np.float32),
            mkpts_ref_s.astype(np.float32),
            method, 3.5, maxIters=1000, confidence=0.999,
        )
        if mask is None or int(mask.sum()) < 4:
            return False

        inl = mask.flatten().astype(bool)

        # 4. Re-project: scale ref keypoints from lr-res back to original ref-res
        mkpts_ref_orig = mkpts_ref_s[inl].copy().astype(np.float32)
        mkpts_ref_orig[:, 0] *= ref_w / lr_w
        mkpts_ref_orig[:, 1] *= ref_h / lr_h

        self._mkpts_lr  = mkpts_lr[inl].astype(np.float32)
        self._mkpts_ref = mkpts_ref_orig
        self._ready = True

        n_in  = int(inl.sum())
        n_raw = len(mkpts_lr)
        print(f"  [FMT] {n_in}/{n_raw} inlier matches  "
              f"(lr {lr_w}x{lr_h}, ref {ref_w}x{ref_h})")
        return True

    def get_ref_crops_batch(
        self,
        batch_pos: List[Tuple[int, int]],
        lr_tile_size: int,
        ref: torch.Tensor,
        ref_tile_size: int,
        lq_h: int,
        lq_w: int,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]]]:
        """
        Compute ref crops for a batch of LR tiles in one call.

        batch_pos     : list of (ty, tx) — top-left corners of LR tiles
        lr_tile_size  : LR tile H = W (pixels)
        ref           : [3, Hr, Wr] original ref tensor (CPU, float32)
        ref_tile_size : target ref crop H = W (pixels; must equal lr_tile_size for the model)
        lq_h, lq_w    : full LR canvas size

        Returns
        -------
        crops  : list of [3, ref_tile_size, ref_tile_size] CPU float32 tensors
        coords : list of (ry, rx, rth, rtw) for visualiser compatibility
        """
        ref_h, ref_w = ref.shape[1], ref.shape[2]
        n = len(batch_pos)
        crops:  List[Optional[torch.Tensor]]          = [None] * n
        coords: List[Tuple[int, int, int, int]]       = [(0, 0, ref_h, ref_w)] * n
        fallback_idx: List[int] = []

        # ── 5 & 6: feature-matched tiles ─────────────────────────────────────
        for i, (ty, tx) in enumerate(batch_pos):
            if self._ready:
                ls = lr_tile_size
                in_tile: np.ndarray = (
                    (self._mkpts_lr[:, 0] >= tx) & (self._mkpts_lr[:, 0] < tx + ls) &
                    (self._mkpts_lr[:, 1] >= ty) & (self._mkpts_lr[:, 1] < ty + ls)
                )
                if in_tile.any():
                    ref_pts = self._mkpts_ref[in_tile]
                    x0, y0, x1, y1 = self._bbox_from_median(
                        ref_pts, ref_h, ref_w, ref_tile_size,
                    )
                    crop = self._crop_or_pad(ref, x0, y0, x1, y1, ref_tile_size)
                    crops[i]  = crop
                    coords[i] = (y0, x0, y1 - y0, x1 - x0)
                    continue

            fallback_idx.append(i)

        # ── 7: proportional fallback — batch resize in one F.interpolate call ─
        if fallback_idx:
            rth = max(1, round(lr_tile_size * ref_h / lq_h))
            rtw = max(1, round(lr_tile_size * ref_w / lq_w))
            need_resize = (rth != ref_tile_size or rtw != ref_tile_size)

            fb_raw:   List[torch.Tensor]              = []
            fb_coord: List[Tuple[int, int, int, int]] = []
            for i in fallback_idx:
                ty, tx = batch_pos[i]
                ry = min(int(ty * ref_h / lq_h), max(0, ref_h - rth))
                rx = min(int(tx * ref_w / lq_w), max(0, ref_w - rtw))
                fb_raw.append(ref[:, ry:ry + rth, rx:rx + rtw].float())
                fb_coord.append((ry, rx, rth, rtw))

            if need_resize:
                batch_fb = torch.stack(fb_raw)          # [M, 3, rth, rtw]
                batch_fb = F.interpolate(
                    batch_fb, size=(ref_tile_size, ref_tile_size),
                    mode='bicubic', align_corners=False,
                ).clamp(0, 1)
                for j, i in enumerate(fallback_idx):
                    crops[i]  = batch_fb[j]
                    coords[i] = fb_coord[j]
            else:
                for j, i in enumerate(fallback_idx):
                    crops[i]  = fb_raw[j].clamp(0, 1)
                    coords[i] = fb_coord[j]

        return crops, coords  # type: ignore[return-value]
