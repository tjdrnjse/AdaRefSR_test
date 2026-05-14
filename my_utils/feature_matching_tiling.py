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

import os
import sys
import colorsys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import List, Optional, Tuple


class FeatureMatchingTiler:
    """XFeat-based ref tile selector for AdaRefSR tiled inference."""

    def __init__(
        self,
        xfeat_project_path: str,
        ransac_threshold: float = 3.5,
        min_inliers: int = 4,
        match_at_ref_resolution: bool = False,
        matcher: str = 'xfeat',
    ):
        """
        xfeat_project_path    : absolute path to the accelerated_features folder.
        ransac_threshold      : RANSAC reprojection error threshold (pixels).
                                Increase (5~8) to tolerate rotation / slight misalignment.
        min_inliers           : minimum post-RANSAC inliers to accept the match.
        match_at_ref_resolution: True  → upsample LQ to ref resolution before matching
                                         (preserves ref feature quality; slower).
                                 False → downsample ref to LQ resolution (original).
        matcher               : 'xfeat'      → XFeat MNN (fast, default)
                                'lighterglue' → XFeat + LighterGlue transformer matcher
                                               (more robust to appearance changes; requires kornia)
        """
        if xfeat_project_path not in sys.path:
            sys.path.insert(0, xfeat_project_path)
        from modules.xfeat import XFeat
        self.xfeat = XFeat()

        self._ransac_threshold = ransac_threshold
        self._min_inliers      = min_inliers
        self._match_at_ref_res = match_at_ref_resolution

        # Validate matcher choice and check kornia availability for lighterglue
        if matcher == 'lighterglue' and not self.xfeat.kornia_available:
            print("[FMT] WARNING: matcher='lighterglue' requires kornia "
                  "(pip install kornia). Falling back to 'xfeat'.")
            matcher = 'xfeat'
        self._matcher = matcher

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

    @staticmethod
    def _tile_starts(total: int, tile_size: int, overlap: int) -> list:
        if total <= tile_size:
            return [0]
        stride = tile_size - overlap
        coords = list(range(0, total - tile_size, stride))
        last = max(0, total - tile_size)
        if not coords or coords[-1] != last:
            coords.append(last)
        return coords

    @staticmethod
    def _hsv_palette(n: int) -> list:
        """Generate n visually distinct BGR colors via HSV cycling."""
        out = []
        for i in range(max(n, 1)):
            h = i / max(n, 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            out.append((int(b * 255), int(g * 255), int(r * 255)))
        return out

    def save_match_visualization(
        self,
        save_path: str,
        lr: torch.Tensor,
        ref: torch.Tensor,
        tile_size_4x: int,
        overlap_4x: int,
        scale: int,
        max_dim: int = 1920,
    ) -> None:
        """
        Save side-by-side feature match visualization.

        Left panel : LQ with tile grid overlay (gray) + matched tiles (colored fill)
                     + LQ keypoints colored by tile.
        Right panel: Ref scaled to LQ resolution + matched ref keypoints
                     + colored ref crop boxes.
        Lines connecting matched LQ ↔ Ref keypoints (sampled, colored by tile).

        Call BEFORE scaling _mkpts_lr by `scale` (i.e., _mkpts_lr must be in
        original LQ coordinate space, same space as the `lr` tensor).
        """
        if not self._ready or self._mkpts_lr is None or self._mkpts_ref is None:
            return

        lr_np  = self._to_uint8(lr)    # [H, W, 3] RGB
        ref_np = self._to_uint8(ref)   # [Hr, Wr, 3] RGB
        lr_h, lr_w   = lr_np.shape[:2]
        ref_h, ref_w = ref_np.shape[:2]

        # Scale ref to LQ display size for side-by-side layout
        ref_disp = cv2.resize(ref_np, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)
        lr_bgr   = cv2.cvtColor(lr_np,   cv2.COLOR_RGB2BGR).copy()
        ref_bgr  = cv2.cvtColor(ref_disp, cv2.COLOR_RGB2BGR).copy()

        # Tile grid in original LQ space (4x tile params divided by scale)
        ts = max(tile_size_4x // scale, 1)
        ov = max(overlap_4x   // scale, 0)
        ys = self._tile_starts(lr_h, ts, ov)
        xs = self._tile_starts(lr_w, ts, ov)

        mkpts_lr  = self._mkpts_lr   # (N,2) x,y in orig LQ space
        mkpts_ref = self._mkpts_ref  # (N,2) x,y in orig ref space

        # Assign each keypoint to the tile it falls in
        tile_to_idx: dict = {}
        for i in range(len(mkpts_lr)):
            px, py = float(mkpts_lr[i, 0]), float(mkpts_lr[i, 1])
            for ty in ys:
                if ty <= py < ty + ts:
                    for tx in xs:
                        if tx <= px < tx + ts:
                            tile_to_idx.setdefault((ty, tx), []).append(i)
                            break
                    break

        # Tiles that have ≥1 matched keypoint, in grid order
        matched_tiles = [(ty, tx) for ty in ys for tx in xs if (ty, tx) in tile_to_idx]
        colors = self._hsv_palette(len(matched_tiles))
        tile_color = {t: c for t, c in zip(matched_tiles, colors)}

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Gray grid for all tiles on LQ
        for ty in ys:
            for tx in xs:
                cv2.rectangle(
                    lr_bgr,
                    (tx, ty),
                    (min(tx + ts, lr_w) - 1, min(ty + ts, lr_h) - 1),
                    (70, 70, 70), 1,
                )

        # Semi-transparent colored fill for matched tiles
        overlay = lr_bgr.copy()
        for (ty, tx), color in tile_color.items():
            cv2.rectangle(
                overlay,
                (tx, ty),
                (min(tx + ts, lr_w) - 1, min(ty + ts, lr_h) - 1),
                color, -1,
            )
        cv2.addWeighted(overlay, 0.22, lr_bgr, 0.78, 0, lr_bgr)

        # Tile index labels on matched tiles
        for k, (ty, tx) in enumerate(matched_tiles):
            color = tile_color[(ty, tx)]
            cx = min(tx + ts // 2, lr_w - 1)
            cy = min(ty + ts // 2, lr_h - 1)
            cv2.putText(lr_bgr, f"T{k}", (max(cx - 8, 0), max(cy + 5, 0)),
                        font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Keypoints on LQ colored by tile
        for (ty, tx), idxs in tile_to_idx.items():
            color = tile_color[(ty, tx)]
            for i in idxs:
                px, py = int(mkpts_lr[i, 0]), int(mkpts_lr[i, 1])
                cv2.circle(lr_bgr, (px, py), 3, color, -1)
                cv2.circle(lr_bgr, (px, py), 4, (255, 255, 255), 1)

        # Ref display scale factors
        sx = lr_w / ref_w
        sy = lr_h / ref_h

        # Keypoints + ref crop boxes on ref display panel
        # Also store ref crop center per tile for the tile-center arrows
        tile_ref_box_center: dict = {}
        for (ty, tx), idxs in tile_to_idx.items():
            color = tile_color[(ty, tx)]
            ref_pts = mkpts_ref[idxs]

            for i in idxs:
                rx = int(mkpts_ref[i, 0] * sx)
                ry = int(mkpts_ref[i, 1] * sy)
                cv2.circle(ref_bgr, (rx, ry), 3, color, -1)
                cv2.circle(ref_bgr, (rx, ry), 4, (255, 255, 255), 1)

            # Ref crop box: proportional size matching actual get_ref_crops_batch behavior
            lq_h_4x = lr_h * scale
            lq_w_4x = lr_w * scale
            vis_crop_size = max(1, round(
                tile_size_4x * ((ref_h / lq_h_4x) * (ref_w / lq_w_4x)) ** 0.5
            ))
            x0, y0, x1, y1 = self._bbox_from_median(ref_pts, ref_h, ref_w, vis_crop_size)
            bx0, by0 = int(x0 * sx), int(y0 * sy)
            bx1, by1 = int(x1 * sx), int(y1 * sy)
            cv2.rectangle(ref_bgr, (bx0, by0), (bx1, by1), color, 2)
            tile_ref_box_center[(ty, tx)] = ((bx0 + bx1) // 2, (by0 + by1) // 2)

        # Combine panels side by side
        combined = np.hstack([lr_bgr, ref_bgr])

        # One arrow per tile: LQ tile center → Ref crop box center
        for (ty, tx), color in tile_color.items():
            lq_cx = min(tx + ts // 2, lr_w - 1)
            lq_cy = min(ty + ts // 2, lr_h - 1)
            ref_cx, ref_cy = tile_ref_box_center[(ty, tx)]
            p1 = (lq_cx, lq_cy)
            p2 = (ref_cx + lr_w, ref_cy)
            cv2.arrowedLine(combined, p1, p2, color, 2, cv2.LINE_AA, tipLength=0.02)

        # Label bar
        bar_h = 24
        bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
        n_tiles  = len(matched_tiles)
        n_inlier = len(mkpts_lr)
        cv2.putText(bar,
                    f"LQ  |  tile grid  ({n_tiles} matched tiles, {n_inlier} inliers)",
                    (6, 17), font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(bar, "Ref (scaled to LQ res)  |  crop boxes",
                    (lr_w + 6, 17), font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        combined = np.vstack([bar, combined])

        # Resize so the longer dimension ≤ max_dim
        ch, cw = combined.shape[:2]
        if max(ch, cw) > max_dim:
            ratio = max_dim / max(ch, cw)
            combined = cv2.resize(
                combined, (int(cw * ratio), int(ch * ratio)),
                interpolation=cv2.INTER_LINEAR,
            )

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, combined)
        print(f"  [FMT] Match visualization saved: {save_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare(self, lr: torch.Tensor, ref: torch.Tensor) -> bool:
        """
        Run the full matching pipeline once per (lr, ref) image pair.

        lr  : [3, H,  W ] in [0,1] — original LQ image (CPU tensor)
        ref : [3, Hr, Wr] in [0,1] — original ref image (CPU tensor)

        Returns True when >= min_inliers RANSAC inlier matches are found.
        Results stored in _mkpts_lr (orig LQ coords) and _mkpts_ref (orig ref coords).
        """
        self._ready = False
        self._mkpts_lr = None
        self._mkpts_ref = None

        lr_h, lr_w   = lr.shape[1],  lr.shape[2]
        ref_h, ref_w = ref.shape[1], ref.shape[2]

        # 1 & 2. Scale Sync + Feature Matching
        if self._match_at_ref_res:
            # Upsample LQ to ref resolution — preserves ref feature quality
            lr_for_match = F.interpolate(
                lr.unsqueeze(0).float(), size=(ref_h, ref_w),
                mode='bicubic', align_corners=False,
            ).squeeze(0).clamp(0, 1)
            img_a = self._to_uint8(lr_for_match)   # LQ at ref resolution
            img_b = self._to_uint8(ref)             # ref at original resolution
        else:
            # Downsample ref to LQ resolution (original behavior)
            ref_scaled = F.interpolate(
                ref.unsqueeze(0).float(), size=(lr_h, lr_w),
                mode='bicubic', align_corners=False,
            ).squeeze(0).clamp(0, 1)
            img_a = self._to_uint8(lr)              # LQ at original resolution
            img_b = self._to_uint8(ref_scaled)      # ref downsampled to LQ resolution

        if self._matcher == 'lighterglue':
            # XFeat detect → LighterGlue transformer match
            out_a = self.xfeat.detectAndCompute(img_a, top_k=4096)[0]
            out_b = self.xfeat.detectAndCompute(img_b, top_k=4096)[0]
            out_a['image_size'] = (img_a.shape[1], img_a.shape[0])  # W, H
            out_b['image_size'] = (img_b.shape[1], img_b.shape[0])
            mkpts_a, mkpts_b, _ = self.xfeat.match_lighterglue(out_a, out_b)
        else:
            # XFeat MNN match (default)
            mkpts_a, mkpts_b = self.xfeat.match_xfeat(img_a, img_b)

        if len(mkpts_a) < 4:
            return False

        # 3. RANSAC outlier removal
        method = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)
        _, mask = cv2.findHomography(
            mkpts_a.astype(np.float32),
            mkpts_b.astype(np.float32),
            method, self._ransac_threshold, maxIters=1000, confidence=0.999,
        )
        if mask is None or int(mask.sum()) < self._min_inliers:
            return False

        inl = mask.flatten().astype(bool)

        # 4. Re-project keypoints to original LQ / ref coordinate spaces
        if self._match_at_ref_res:
            # mkpts_a is in ref-res space → scale down to original LQ coords
            mkpts_lr_orig = mkpts_a[inl].copy().astype(np.float32)
            mkpts_lr_orig[:, 0] *= lr_w / ref_w
            mkpts_lr_orig[:, 1] *= lr_h / ref_h
            # mkpts_b is already in original ref coords
            mkpts_ref_orig = mkpts_b[inl].astype(np.float32)
        else:
            # mkpts_a is already in original LQ coords
            mkpts_lr_orig = mkpts_a[inl].astype(np.float32)
            # mkpts_b is in LQ-res space → scale up to original ref coords
            mkpts_ref_orig = mkpts_b[inl].copy().astype(np.float32)
            mkpts_ref_orig[:, 0] *= ref_w / lr_w
            mkpts_ref_orig[:, 1] *= ref_h / lr_h

        self._mkpts_lr  = mkpts_lr_orig
        self._mkpts_ref = mkpts_ref_orig
        self._ready = True

        n_in  = int(inl.sum())
        n_raw = len(mkpts_a)
        mode  = f"ref-res {ref_w}x{ref_h}" if self._match_at_ref_res \
                else f"lq-res {lr_w}x{lr_h}"
        print(f"  [FMT] {n_in}/{n_raw} inliers  "
              f"(matcher={self._matcher}, match@{mode}, "
              f"threshold={self._ransac_threshold}, min_inliers={self._min_inliers})")
        return True

    def get_ref_crops_batch(
        self,
        batch_pos: List[Tuple[int, int]],
        lr_tile_size: int,
        ref: torch.Tensor,
        ref_tile_size: int,
        lq_h: int,
        lq_w: int,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]], List[bool]]:
        """
        Compute ref crops for a batch of LR tiles in one call.

        batch_pos     : list of (ty, tx) — top-left corners of LR tiles
        lr_tile_size  : LR tile H = W (pixels)
        ref           : [3, Hr, Wr] original ref tensor (CPU, float32)
        ref_tile_size : target ref crop H = W (pixels; must equal lr_tile_size for the model)
        lq_h, lq_w    : full LR canvas size

        Returns
        -------
        crops         : list of [3, ref_tile_size, ref_tile_size] CPU float32 tensors
        coords        : list of (ry, rx, rth, rtw) for visualiser compatibility
        matched_flags : list of bool — True if tile got an FMT match, False if proportional fallback
        """
        ref_h, ref_w = ref.shape[1], ref.shape[2]
        n = len(batch_pos)
        crops:         List[Optional[torch.Tensor]]    = [None] * n
        coords:        List[Tuple[int, int, int, int]] = [(0, 0, ref_h, ref_w)] * n
        matched_flags: List[bool]                      = [True] * n   # assume matched; fallback marks False
        fallback_idx:  List[int] = []

        # ── 5 & 6: feature-matched tiles ─────────────────────────────────────
        # Proportional crop size: same spatial fraction as lr_tile_size / lq_h.
        # Geometric mean of h/w scale factors handles differing aspect ratios.
        ref_crop_size = max(1, round(
            ref_tile_size * ((ref_h / lq_h) * (ref_w / lq_w)) ** 0.5
        ))
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
                        ref_pts, ref_h, ref_w, ref_crop_size,
                    )
                    crop = self._crop_or_pad(ref, x0, y0, x1, y1, ref_tile_size)
                    crops[i]  = crop
                    coords[i] = (y0, x0, y1 - y0, x1 - x0)
                    continue

            fallback_idx.append(i)
            matched_flags[i] = False   # no keypoints in tile → proportional fallback

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

        return crops, coords, matched_flags  # type: ignore[return-value]
