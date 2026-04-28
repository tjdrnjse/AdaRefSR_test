"""
visualize_aicg.py  –  Ada-RefSR "Trust but Verify" 레퍼런스 히트맵 시각화
==========================================================================

논문 수식 대응 (Wang et al., "Trust but Verify"):

  Eq.1  Q    = H_src W_Q,  K = H_ref W_K,  V = H_ref W_V
  Eq.2  A_RA = Softmax(QK^T / √d)                    [B*h, L_q, L_ref]
        H_out = ZeroLinear(A_RA @ V) + H_src
  Eq.3  S    = T_S W_K                                [B*h, M, C']
  Eq.4  Ksum = Softmax(S K^T / √d) @ K               [B*h, M, C']
  Eq.5  Smap = Softmax(Q Ksum^T / √d)                [B*h, L_q, M]
  Eq.6  G    = σ( mean_{j=1..M, heads}(Smap) )       [B, L_q]
  Eq.7  H_out = ZeroLinear( G ⊙ RA(H_src, H_ref) ) + H_src

시각화 3종:
  Map1 Trust    : ROI_mask.T  @ A_RA_avg  → [L_ref] → Ref 캔버스 투영
  Map2 Verify   : (G * ROI)               → LQ 공간 1D 벡터 (Attention 투영 없음)
  Map3 Combined : (ROI * G).T @ A_RA_avg  → [L_ref] → Ref 캔버스 투영
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================================
# 1. 어텐션 레벨 투영 함수
# ============================================================================

def compute_aicg_maps(
    query:   Optional[torch.Tensor],   # [h, L_q,   C'] – None if probs provided
    key:     Optional[torch.Tensor],   # [h, L_ref,  C'] – None if probs provided
    gate:    torch.Tensor,             # [1, L_q]         G (Eq.6, post-intervention)
    roi_vec: torch.Tensor,             # [L_q]            ROI 마스크 (0~1)
    scale:   float,                    # 1/√C' – unused when probs provided
    probs:   Optional[torch.Tensor] = None,  # [h, L_q, L_ref] – pre-computed A_RA
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A_RA(Eq.2)를 헤드별 계산 후 즉시 [L_ref]로 투영·삭제 (Map1, Map3).
    Map2 는 Attention 투영 없이 LQ 공간의 (G * ROI) 를 2D로 반환.

    probs 가 주어지면 A_RA 재연산을 생략하고 steerer 캐시 값을 그대로 사용.
    probs 가 None 이면 (query, key, scale) 로 fallback 재연산.

    Returns:
      map1 : [L_ref]   Trust    – ROI → Ref 공간
      map2 : [L_q]     Verify   – G×ROI 1D (stitch 시 2D 변환)
      map3 : [L_ref]   Combined – (ROI×G) → Ref 공간
    """
    # 디바이스 결정: probs/gate 기준 (query 없어도 동작)
    if probs is not None:
        device    = probs.device
        num_heads = probs.shape[0]   # h
        L_q       = probs.shape[1]   # L_q
        L_ref     = probs.shape[2]   # L_ref
    else:
        assert query is not None and key is not None
        device    = query.device
        num_heads = query.shape[0]
        L_q       = query.shape[1]
        L_ref     = key.shape[1]

    ls_q = int(math.sqrt(L_q))

    roi = roi_vec.float().to(device)
    if roi.shape[0] != L_q:
        ls_src = int(math.sqrt(roi.shape[0]))
        roi = F.interpolate(
            roi.reshape(1, 1, ls_src, ls_src),
            size=(ls_q, ls_q),
            mode='bilinear',
            align_corners=False,
        ).reshape(-1).to(device)

    g     = gate[0].float().to(device)  # [L_q]
    roi_g = roi * g

    map2 = roi_g  # [L_q] – stitcher가 2D로 변환

    acc1 = torch.zeros(L_ref, dtype=torch.float32, device=device)
    acc3 = torch.zeros(L_ref, dtype=torch.float32, device=device)

    if probs is not None:
        # ── Steerer 캐시 경로: A_RA 재연산 없이 직접 투영 ──────────────────
        for h_idx in range(num_heads):
            # a_ra_h: [L_q, L_ref] – Trust-intervened attention probabilities
            a_ra_h = probs[h_idx].float().to(device)
            acc1 += roi   @ a_ra_h
            acc3 += roi_g @ a_ra_h
            del a_ra_h
    else:
        # ── Fallback 경로: (query, key) 로 A_RA 재연산 ─────────────────────
        q32 = query.float()
        k32 = key.float()
        for h_idx in range(num_heads):
            a_ra_h = torch.softmax(q32[h_idx] @ k32[h_idx].T * scale, dim=-1)
            acc1 += roi   @ a_ra_h
            acc3 += roi_g @ a_ra_h
            del a_ra_h

    return acc1 / num_heads, map2, acc3 / num_heads


# ============================================================================
# 2. 타일별 레이어 누적 저장소
# ============================================================================

class _LayerAccumulator:
    """한 타일 forward의 여러 어텐션 레이어 결과를 누적.

    m1, m3 : [L_ref]      Ref 공간 1D 벡터 (레이어간 bilinear 리사이즈 후 누적)
    m2     : [ls_q, ls_q] LQ 공간 2D 맵    (레이어간 bilinear 리사이즈 후 누적)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.m1: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None
        self.m3: Optional[torch.Tensor] = None
        self.n:  int = 0

    @staticmethod
    def _resize1d(v: torch.Tensor, target_L: int) -> torch.Tensor:
        """[L] → 2D bilinear resize → [target_L] (공간 경계 보존)."""
        if v.shape[0] == target_L:
            return v
        src_ls = int(math.sqrt(v.shape[0]))
        tgt_ls = int(math.sqrt(target_L))
        return F.interpolate(
            v.float().reshape(1, 1, src_ls, src_ls),
            size=(tgt_ls, tgt_ls),
            mode='bilinear',
            align_corners=False,
        ).reshape(-1)

    def update(self, m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor):
        if self.m1 is None:
            self.m1 = m1.clone()
            self.m2 = m2.clone()
            self.m3 = m3.clone()
        else:
            self.m1 += self._resize1d(m1, self.m1.shape[0])
            self.m2 += self._resize1d(m2, self.m2.shape[0])
            self.m3 += self._resize1d(m3, self.m3.shape[0])
        self.n += 1

    def mean(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = max(self.n, 1)
        return self.m1 / c, self.m2 / c, self.m3 / c


# ============================================================================
# 3. AICGVisualizer – 배치 단위 캡처 + 글로벌 캔버스 관리
# ============================================================================

class AICGVisualizer:
    """
    타일링 환경에서 AICG 시각화 맵을 글로벌 캔버스에 누적.

    Map1, Map3 : Ref 해상도 캔버스 (ref_h × ref_w)
    Map2       : LQ 해상도 캔버스  (lq_h  × lq_w)  – Gate×ROI LQ 공간

    demo_tiled.py 통합 예시:
    ─────────────────────────────────────────────────────────
        viz = AICGVisualizer(net_sr.unet, roi_mask,
                             ref_h, ref_w, lq_h, lq_w, tile_size)

        for batch_start in range(0, n_tiles, batch_sz):
            tile_coords = [...]   # (ty,tx,ry,rx,rth,rtw) per tile

            with viz.capture_batch(tile_coords, lq_h, lq_w):
                predictions = infer_batch(x_src_b, x_ref_b, ...)

        viz.finalize(np.array(ref_img), np.array(lq_img), "output_vis.png")
    ─────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        unet,
        roi_mask:      Optional[np.ndarray],   # [lq_h, lq_w] float32 0~1 or None
        ref_h:         int,
        ref_w:         int,
        lq_h:          int,
        lq_w:          int,
        tile_size:     int = 512,
        fusion_blocks: str = "full",
        steerer=None,  # Optional[AICGSteerer] – inject to use cached probs/gate
    ):
        self.unet      = unet
        self.roi_mask  = roi_mask
        self.steerer   = steerer   # if provided, hook reads steerer._last_probs/_last_gate
        self.ref_h     = ref_h
        self.ref_w     = ref_w
        self.lq_h      = lq_h
        self.lq_w      = lq_w
        self.tile_size = tile_size

        # Map1, Map3 → Ref 공간 캔버스
        self.canvas1  = torch.zeros(ref_h, ref_w)
        self.canvas3  = torch.zeros(ref_h, ref_w)
        self.cnt_ref  = torch.zeros(ref_h, ref_w)

        # Map2 → LQ 공간 캔버스
        self.canvas2  = torch.zeros(lq_h, lq_w)
        self.cnt_lq   = torch.zeros(lq_h, lq_w)

        self._hook_handles:  list                              = []
        self._batch_accs:    Optional[List[_LayerAccumulator]] = None
        self._batch_roi_pix: Optional[List]                    = None

        self._attn_refs = self._collect_attn_refs(fusion_blocks)
        if not self._attn_refs:
            print("[AICGVisualizer WARNING] attn_ref 모듈을 찾지 못했습니다.")

    # ── 모듈 수집 ─────────────────────────────────────────────────────────────

    def _collect_attn_refs(self, fusion_blocks: str):
        if self.unet is None:
            return []
        try:
            from diffusers.models.attention import BasicTransformerBlock
        except ImportError:
            return []

        def dfs(m):
            r = [m]
            for c in m.children():
                r += dfs(c)
            return r

        candidates = (
            dfs(self.unet.mid_block) + dfs(self.unet.up_blocks)
            if fusion_blocks == "midup" else dfs(self.unet)
        )
        return [
            m.attn_ref for m in candidates
            if isinstance(m, BasicTransformerBlock) and hasattr(m, 'attn_ref')
        ]

    # ── Hook 설치 / 제거 ──────────────────────────────────────────────────────

    def _install(self):
        batch_accs = self._batch_accs
        batch_roi  = self._batch_roi_pix
        ts         = self.tile_size
        steerer    = self.steerer   # may be None

        for attn_ref in self._attn_refs:

            def _make_hook(ar):
                def hook(module, args, kwargs, output):

                    # ── 경로 1: Steerer 캐시 읽기 (Trust/Verify 개입 결과 반영) ──
                    # SteeredReferenceAttnProcessor 가 이 forward 직전에 실행되고
                    # probs [B*h, L_q, L_ref] / gate [B, L_q] 를 CPU 에 캐싱해 둔 상태.
                    if (
                        steerer is not None
                        and steerer.capture_for_viz
                        and steerer._last_probs is not None
                        and steerer._last_gate  is not None
                    ):
                        # probs: [B*h, L_q, L_ref] – Trust-intervened attention map (cpu)
                        # gate : [B,   L_q]         – Verify-intervened gate         (cpu)
                        cached_probs = steerer._last_probs
                        cached_gate  = steerer._last_gate
                        B_val   = cached_gate.shape[0]
                        h_eff   = cached_probs.shape[0] // B_val
                        L_q_act = cached_probs.shape[1]
                        ls_act  = int(math.sqrt(L_q_act))

                        for i in range(min(B_val, len(batch_accs))):
                            h0, h1   = i * h_eff, (i + 1) * h_eff
                            probs_i  = cached_probs[h0:h1]       # [h, L_q, L_ref]
                            gate_i   = cached_gate[i:i + 1]      # [1, L_q]

                            roi_pix = batch_roi[i]
                            if roi_pix is not None:
                                roi_vec = F.interpolate(
                                    roi_pix.unsqueeze(0).unsqueeze(0).float(),
                                    size=(ls_act, ls_act),
                                    mode='area',
                                ).squeeze().reshape(-1)           # cpu
                            else:
                                roi_vec = torch.ones(L_q_act)

                            m1, m2, m3 = compute_aicg_maps(
                                query=None, key=None,
                                gate=gate_i, roi_vec=roi_vec,
                                scale=0.0, probs=probs_i,
                            )
                            batch_accs[i].update(m1, m2, m3)
                        return  # steerer 경로 완료 – fallback 생략

                    # ── 경로 2: Fallback – 순정 수식으로 재연산 ──────────────────
                    # (steerer 없음 또는 capture_for_viz=False 인 경우)
                    hs  = args[0] if len(args) > 0 else kwargs.get('hidden_states')
                    enc = kwargs.get('encoder_hidden_states')
                    if hs is None or enc is None:
                        return

                    B_val = hs.shape[0]
                    if hs.ndim == 4:
                        hs = hs.view(B_val, hs.shape[1], -1).transpose(1, 2)
                    enc = enc.to(hs.dtype)
                    if enc.ndim == 4:
                        enc = enc.view(B_val, enc.shape[1], -1).transpose(1, 2)

                    with torch.no_grad():
                        q = ar.head_to_batch_dim(ar.to_q(hs))
                        k = ar.head_to_batch_dim(ar.to_k(enc))

                        ts_tok = ar.learnable_token.expand(B_val, -1, -1)
                        ts_k   = ar.head_to_batch_dim(ts_tok)
                        s_ref  = torch.softmax(
                            torch.bmm(ts_k, k.transpose(-1, -2)) * ar.scale, dim=-1)
                        ksum   = torch.bmm(s_ref, k)
                        smap_logits = torch.bmm(q, ksum.transpose(-1, -2)) * ar.scale
                        gate        = torch.sigmoid(
                            ar.batch_to_head_dim(smap_logits).mean(dim=-1))  # [B, L_q]

                        actual_heads = q.shape[0] // B_val
                        L_q_actual   = q.shape[1]
                        ls_actual    = int(math.sqrt(L_q_actual))

                        for i in range(min(B_val, len(batch_accs))):
                            h0, h1 = i * actual_heads, (i + 1) * actual_heads
                            q_i    = q   [h0:h1]
                            k_i    = k   [h0:h1]
                            gate_i = gate[i:i + 1]

                            roi_pix = batch_roi[i]
                            if roi_pix is not None:
                                roi_vec = F.interpolate(
                                    roi_pix.unsqueeze(0).unsqueeze(0).float(),
                                    size=(ls_actual, ls_actual),
                                    mode='area',
                                ).squeeze().reshape(-1).to(q.device)
                            else:
                                roi_vec = torch.ones(L_q_actual, device=q.device)

                            m1, m2, m3 = compute_aicg_maps(
                                query=q_i, key=k_i,
                                gate=gate_i, roi_vec=roi_vec,
                                scale=ar.scale,
                            )
                            batch_accs[i].update(m1, m2, m3)

                return hook

            handle = attn_ref.register_forward_hook(
                _make_hook(attn_ref), with_kwargs=True
            )
            self._hook_handles.append(handle)

    def _remove(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ── 글로벌 캔버스 스티칭 ─────────────────────────────────────────────────

    def _stitch_one(
        self,
        acc:     _LayerAccumulator,
        ty:      int, tx:      int,   # LQ 공간 타일 좌표
        ref_ry:  int, ref_rx:  int,   # Ref 공간 타일 좌표
        rth:     int, rtw:     int,   # Ref 타일 높이·너비
    ):
        """단일 타일 레이어 평균 맵을 각 캔버스에 누적."""
        if acc.n == 0:
            return

        m1, m2, m3 = acc.mean()

        # ── Map1, Map3 → Ref 캔버스 ──────────────────────────────────────────
        def to_2d_ref(vec: torch.Tensor) -> torch.Tensor:
            l  = vec.shape[0]
            ls = int(math.sqrt(l))
            if ls * ls != l:
                ls = self.tile_size // 8
                l  = ls * ls
            v  = vec.cpu().float()[:l].reshape(1, 1, ls, ls)
            mn, mx = v.min(), v.max()
            v  = (v - mn) / (mx - mn + 1e-8)
            return F.interpolate(
                v, size=(rth, rtw), mode='bilinear', align_corners=False
            ).squeeze()

        r0, r1 = ref_ry, ref_ry + rth
        c0, c1 = ref_rx, ref_rx + rtw
        self.canvas1[r0:r1, c0:c1] += to_2d_ref(m1)
        self.canvas3[r0:r1, c0:c1] += to_2d_ref(m3)
        self.cnt_ref[r0:r1, c0:c1] += 1.0

        # ── Map2 → LQ 캔버스 (1D → 2D 변환 후 스티칭) ───────────────────────
        ts = self.tile_size
        th = min(ts, self.lq_h - ty)
        tw = min(ts, self.lq_w - tx)
        if th > 0 and tw > 0:
            l_q  = m2.shape[0]
            ls_q = int(math.sqrt(l_q))
            m2_up = F.interpolate(
                m2.cpu().float().reshape(1, 1, ls_q, ls_q),
                size=(ts, ts),
                mode='bilinear', align_corners=False,
            ).squeeze()  # [ts, ts]
            self.canvas2[ty:ty + th, tx:tx + tw] += m2_up[:th, :tw]
            self.cnt_lq [ty:ty + th, tx:tx + tw] += 1.0

    # ── 배치 컨텍스트 매니저 (메인 인터페이스) ───────────────────────────────

    @contextmanager
    def capture_batch(
        self,
        tile_coords: List[Tuple[int, int, int, int, int, int]],
        # [(ty, tx, ref_ry, ref_rx, rth, rtw), ...]
        lq_h: int,
        lq_w: int,
    ):
        """배치 내 타일 전체를 한 번의 infer_batch로 처리하면서 시각화."""
        ts = self.tile_size
        B  = len(tile_coords)

        self._batch_roi_pix = []
        for (ty, tx, *_) in tile_coords:
            if self.roi_mask is not None:
                crop = self.roi_mask[ty:ty + ts, tx:tx + ts].astype(np.float32)
                if crop.shape[0] != ts or crop.shape[1] != ts:
                    padded = np.zeros((ts, ts), dtype=np.float32)
                    padded[:crop.shape[0], :crop.shape[1]] = crop
                    crop = padded
                self._batch_roi_pix.append(torch.from_numpy(crop))
            else:
                self._batch_roi_pix.append(None)

        self._batch_accs = [_LayerAccumulator() for _ in range(B)]

        self._install()
        try:
            yield
        finally:
            self._remove()
            for i, (ty, tx, ry, rx, rth, rtw) in enumerate(tile_coords):
                self._stitch_one(self._batch_accs[i], ty, tx, ry, rx, rth, rtw)
            self._batch_accs    = None
            self._batch_roi_pix = None

    @contextmanager
    def capture_tile(
        self,
        ty: int, tx: int,
        ref_ry: int, ref_rx: int,
        rth: int, rtw: int,
        lq_h: int, lq_w: int,
    ):
        """단일 타일용 편의 래퍼 (capture_batch 위임)."""
        with self.capture_batch(
            [(ty, tx, ref_ry, ref_rx, rth, rtw)], lq_h, lq_w
        ):
            yield

    # ── 최종 출력 ─────────────────────────────────────────────────────────────

    def finalize(
        self,
        ref_img:     np.ndarray,   # [ref_h, ref_w, 3]  Map1·Map3 블렌딩 배경
        lq_img:      np.ndarray,   # [lq_h,  lq_w,  3]  Map2 블렌딩 배경
        output_path: str,
        colormap:    str = 'jet',
    ):
        """3개 글로벌 맵을 각 배경 이미지와 블렌딩 → 가로 병합 → PNG 저장.

        Map1, Map3 는 ref_img 와, Map2 는 lq_img 와 블렌딩.
        높이가 다를 경우 Map2 를 ref_img 높이에 맞춰 리사이즈 후 연결.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cnt_ref = self.cnt_ref.clamp(min=1.0)
        cnt_lq  = self.cnt_lq.clamp(min=1.0)
        g1 = (self.canvas1 / cnt_ref).numpy()
        g2 = (self.canvas2 / cnt_lq ).numpy()
        g3 = (self.canvas3 / cnt_ref).numpy()
        cm_fn = plt.get_cmap(colormap)

        def blend(hm: np.ndarray, base: np.ndarray, is_gate: bool = False) -> np.ndarray:
            if is_gate:
                hm_n = hm.clip(0, 1)
            else:
                mn, mx = hm.min(), hm.max()
                hm_n   = (hm - mn) / (mx - mn + 1e-8)
            hm_rgb = (cm_fn(hm_n) * 255).astype(np.uint8)[:, :, :3]
            return ((base.astype(np.float32) * 0.5 +
                     hm_rgb.astype(np.float32) * 0.5)
                    .clip(0, 255).astype(np.uint8))

        b1 = blend(g1, ref_img, is_gate=False)   # [ref_h, ref_w, 3]
        b2 = blend(g2, lq_img,  is_gate=True)    # [lq_h,  lq_w,  3]
        b3 = blend(g3, ref_img, is_gate=False)   # [ref_h, ref_w, 3]

        # 높이를 ref_h 로 통일
        target_h = ref_img.shape[0]
        if b2.shape[0] != target_h:
            new_w = max(1, round(b2.shape[1] * target_h / b2.shape[0]))
            b2 = np.array(
                Image.fromarray(b2).resize((new_w, target_h), Image.BILINEAR)
            )

        out = np.concatenate([b1, b2, b3], axis=1)

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(out).save(output_path)
        print(f"[AICGVisualizer] 저장 완료: {output_path}  "
              f"({out.shape[1]}x{out.shape[0]} px)")
