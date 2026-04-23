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
  Map1 Trust    : ROI_mask.T  @ A_RA_avg  [L_ref]  – ROI 픽셀이 Ref 어디를 참조했는가
  Map2 Verify   : G.T         @ A_RA_avg  [L_ref]  – Gate 가중치를 Ref 공간으로 역투영
  Map3 Combined : (ROI * G).T @ A_RA_avg  [L_ref]  – 최종 복원 기여 위치

호환성:
  mmcv_compat.py / migrate_mmcv.py 와 독립적으로 동작.
  demo_tiled.py 에서 import 전에 mmcv_compat 이 이미 로드되어 있다고 가정.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================================
# 1. 어텐션 레벨 투영 함수
# ============================================================================

def compute_aicg_maps(
    query:     torch.Tensor,   # [B*h, L_q,   C']  –  Q (Eq.1)
    key:       torch.Tensor,   # [B*h, L_ref,  C']  –  K (Eq.1)
    gate:      torch.Tensor,   # [B,   L_q]          –  G (Eq.6)
    roi_vec:   torch.Tensor,   # [L_q]               –  ROI 마스크 (토큰 공간, 0~1)
    scale:     float,          # 1/√C'  (attn.scale)
    B:         int,            # 배치 크기
    num_heads: int,            # 어텐션 헤드 수
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A_RA 행렬(Eq.2)을 헤드별로 계산한 뒤 즉시 [L_ref] 벡터로 투영하고 삭제.
    행렬 자체([B*h, L_q, L_ref])는 캔버스에 저장하지 않음(메모리 절약).

    연산 단계별 Shape:
      q32, k32       : [B*h, L_q/L_ref, C']         입력 Q/K (float32 업캐스트)
      a_ra_h (헤드 h): [L_q, L_ref]                  A_RA_h = Softmax(Q_h K_h^T/√d) – Eq.2
      acc1/acc2/acc3 : [L_ref]                        헤드 평균 누적

    Returns:
      (map1, map2, map3)  각각 shape [L_ref],  CPU float32
    """
    device = query.device
    L_ref  = key.shape[1]

    roi   = roi_vec.float().to(device)   # [L_q]
    g     = gate[0].float().to(device)  # [L_q]  (B=1 슬라이스)
    roi_g = roi * g                      # [L_q]  ROI × G (Eq.6 결합, Map3용)

    acc1 = torch.zeros(L_ref, dtype=torch.float32, device=device)
    acc2 = torch.zeros(L_ref, dtype=torch.float32, device=device)
    acc3 = torch.zeros(L_ref, dtype=torch.float32, device=device)

    q32 = query.float()   # [B*h, L_q,   C']
    k32 = key.float()     # [B*h, L_ref, C']

    for h_idx in range(B * num_heads):
        q_h = q32[h_idx]   # [L_q,   C']
        k_h = k32[h_idx]   # [L_ref, C']

        # A_RA_h = Softmax(Q_h K_h^T / √d)  [L_q, L_ref]  – Eq.2
        # 64 MB/head(float32, 4096×4096): 즉시 투영 후 del 로 해제
        a_ra_h = torch.softmax(q_h @ k_h.T * scale, dim=-1)   # [L_q, L_ref]

        # Map1  Trust    : ROI.T      @ A_RA_h  →  [L_ref]
        acc1 += roi   @ a_ra_h
        # Map2  Verify   : G.T        @ A_RA_h  →  [L_ref]
        acc2 += g     @ a_ra_h
        # Map3  Combined : (ROI×G).T  @ A_RA_h  →  [L_ref]
        acc3 += roi_g @ a_ra_h

        del a_ra_h   # 행렬 즉시 해제 (요구사항: 캔버스에 직접 저장 금지)

    n = B * num_heads
    return acc1 / n, acc2 / n, acc3 / n   # 각각 [L_ref]


# ============================================================================
# 2. 타일별 레이어 누적 저장소
# ============================================================================

class _LayerAccumulator:
    """한 타일 forward 에서 여러 어텐션 레이어의 결과를 누적."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.m1: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None
        self.m3: Optional[torch.Tensor] = None
        self.n:  int = 0

    def update(self, m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor):
        if self.m1 is None:
            self.m1, self.m2, self.m3 = m1.clone(), m2.clone(), m3.clone()
        else:
            self.m1 += m1; self.m2 += m2; self.m3 += m3
        self.n += 1

    def mean(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = max(self.n, 1)
        return self.m1 / c, self.m2 / c, self.m3 / c


# ============================================================================
# 3. AICGVisualizer – 글로벌 캔버스 관리 및 타일 스티칭
# ============================================================================

class AICGVisualizer:
    """
    타일링 환경에서 AICG 시각화 맵을 글로벌 캔버스에 누적.

    demo_tiled.py 통합 예시:
    ──────────────────────────────────────────────────────────────────
        viz = AICGVisualizer(net_sr.unet, roi_mask=None,
                             ref_h=ref_h, ref_w=ref_w,
                             tile_size=tile_size)

        for batch_start in range(0, n_tiles, batch_sz):
            batch_pos = all_tiles[batch_start : batch_start + batch_sz]
            for (ty, tx) in batch_pos:
                ry  = int(ty * ref_h / lq_h)
                rx  = int(tx * ref_w / lq_w)
                rth = max(1, round(tile_size * ref_h / lq_h))
                rtw = max(1, round(tile_size * ref_w / lq_w))
                with viz.capture_tile(ty, tx, ry, rx, rth, rtw, lq_h, lq_w):
                    predictions = infer_batch(...)

        viz.finalize(np.array(ref_img), "output_heatmap.png")
    ──────────────────────────────────────────────────────────────────

    Args:
        unet          : GenModel.unet  (attn_ref 포함한 UNet)
        roi_mask      : np.ndarray [lq_h, lq_w] float32 0~1 또는 None (전체)
        ref_h / ref_w : 전체 Ref 이미지 픽셀 해상도
        tile_size     : 타일 한 변 픽셀 크기
        num_heads     : UNet 어텐션 헤드 수 (기본 8)
        fusion_blocks : "full" 또는 "midup"
    """

    def __init__(
        self,
        unet,
        roi_mask:       Optional[np.ndarray],
        ref_h:          int,
        ref_w:          int,
        tile_size:      int = 512,
        num_heads:      int = 8,
        fusion_blocks:  str = "full",
    ):
        self.unet        = unet
        self.roi_mask    = roi_mask
        self.ref_h       = ref_h
        self.ref_w       = ref_w
        self.tile_size   = tile_size
        self.num_heads   = num_heads
        self.latent_sz   = tile_size // 8   # VAE 다운샘플링 배율

        # 전역 캔버스 3개 + 카운팅 맵  [ref_h, ref_w]  CPU float32
        self.canvas1 = torch.zeros(ref_h, ref_w)   # Trust    (Map1)
        self.canvas2 = torch.zeros(ref_h, ref_w)   # Verify   (Map2)
        self.canvas3 = torch.zeros(ref_h, ref_w)   # Combined (Map3)
        self.cnt     = torch.zeros(ref_h, ref_w)   # 타일 누적 횟수 (경계 평균화)

        self._acc:          _LayerAccumulator       = _LayerAccumulator()
        self._hook_handles: list                    = []
        self._cur_roi:      Optional[torch.Tensor]  = None

        self._attn_refs = self._collect_attn_refs(fusion_blocks)
        if not self._attn_refs:
            print("[AICGVisualizer WARNING] attn_ref 모듈을 찾지 못했습니다. "
                  "AICG 시각화 결과가 비어 있을 수 있습니다.")

    # ── 모듈 수집 ─────────────────────────────────────────────────────────────

    def _collect_attn_refs(self, fusion_blocks: str):
        try:
            from diffusers.models.attention import BasicTransformerBlock
        except ImportError:
            return []

        def dfs(m):
            r = [m]
            for c in m.children():
                r += dfs(c)
            return r

        if fusion_blocks == "midup":
            candidates = dfs(self.unet.mid_block) + dfs(self.unet.up_blocks)
        else:
            candidates = dfs(self.unet)

        return [
            m.attn_ref for m in candidates
            if isinstance(m, BasicTransformerBlock) and hasattr(m, 'attn_ref')
        ]

    # ── Hook 설치 / 제거 ──────────────────────────────────────────────────────

    def _install(self):
        acc       = self._acc
        num_heads = self.num_heads
        get_roi   = lambda: self._cur_roi

        for attn_ref in self._attn_refs:

            def _make_hook(ar):
                def hook(module, args, kwargs, output):
                    # args[0]                        = hidden_states  H_src (Eq.1)
                    # kwargs['encoder_hidden_states'] = H_ref         (Eq.1)
                    hs  = args[0] if len(args) > 0 else kwargs.get('hidden_states')
                    enc = kwargs.get('encoder_hidden_states')
                    if hs is None or enc is None:
                        return

                    B_val = hs.shape[0]
                    if hs.ndim == 4:
                        hs = hs.view(B_val, hs.shape[1], -1).transpose(1, 2)  # [B,L,C]
                    enc = enc.to(hs.dtype)
                    if enc.ndim == 4:
                        enc = enc.view(B_val, enc.shape[1], -1).transpose(1, 2)

                    with torch.no_grad():
                        # Q = H_src W_Q  [B*h, L_q,   C']  – Eq.1
                        # K = H_ref W_K  [B*h, L_ref, C']  – Eq.1
                        q = ar.head_to_batch_dim(ar.to_q(hs))    # [B*h, L_q,   C']
                        k = ar.head_to_batch_dim(ar.to_k(enc))   # [B*h, L_ref, C']

                        # ── G 재현 (Eq.3 ~ Eq.6) ─────────────────────────
                        # Eq.3: S = T_S W_K
                        ts   = ar.learnable_token.expand(B_val, -1, -1)  # [B,  M, C]
                        ts_k = ar.head_to_batch_dim(ts)                   # [B*h,M, C']

                        # Eq.4: Ksum = Softmax(S K^T/√d) K
                        s_ref = torch.softmax(
                            torch.bmm(ts_k, k.transpose(-1, -2)) * ar.scale,
                            dim=-1,
                        )                                                  # [B*h, M, L_ref]
                        ksum = torch.bmm(s_ref, k)                        # [B*h, M, C']

                        # Eq.5: Smap = Softmax(Q Ksum^T/√d)
                        smap = torch.softmax(
                            torch.bmm(q, ksum.transpose(-1, -2)) * ar.scale,
                            dim=-1,
                        )                                                  # [B*h, L_q, M]

                        # Eq.6: G = σ( mean_{M,heads}(Smap) )
                        # batch_to_head_dim: [B*h,L_q,M] → [B,L_q,M*h]
                        gate = torch.sigmoid(
                            ar.batch_to_head_dim(smap).mean(dim=-1)
                        )                                                  # [B, L_q]

                        # ROI 준비 (없으면 전체 1)
                        roi = get_roi()
                        if roi is None:
                            roi = torch.ones(q.shape[1], device=q.device)

                        # Map 계산 (A_RA 즉시 해제)
                        m1, m2, m3 = compute_aicg_maps(
                            q, k, gate, roi, ar.scale, B_val, num_heads
                        )
                        acc.update(m1, m2, m3)

                return hook

            handle = attn_ref.register_forward_hook(
                _make_hook(attn_ref), with_kwargs=True
            )
            self._hook_handles.append(handle)

    def _remove(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ── 타일별 컨텍스트 매니저 ───────────────────────────────────────────────

    @contextmanager
    def capture_tile(
        self,
        ty:     int, tx:     int,   # LQ SR canvas 기준 타일 좌상단 좌표 (픽셀)
        ref_ry: int, ref_rx: int,   # Ref 이미지 기준 타일 좌상단 좌표 (픽셀)
        rth:    int, rtw:    int,   # Ref 타일 크기 (픽셀)
        lq_h:   int, lq_w:  int,   # 전체 LQ SR canvas 크기 (픽셀)
    ):
        """
        타일 1개 인퍼런스를 감싸는 컨텍스트.
        __enter__: hook 설치 + ROI 준비
        __exit__ : hook 제거 + 글로벌 캔버스 누적
        """
        ts     = self.tile_size
        ls     = self.latent_sz
        device = next(self.unet.parameters()).device

        # ROI: LQ 픽셀 마스크 → 타일 크롭 → latent 해상도 다운샘플 → [L_q]
        if self.roi_mask is not None:
            roi_crop = torch.from_numpy(
                self.roi_mask[ty:ty + ts, tx:tx + ts].astype(np.float32)
            ).unsqueeze(0).unsqueeze(0)                              # [1,1,ts,ts]
            roi_lat  = F.interpolate(roi_crop, size=(ls, ls), mode='area').squeeze()
            self._cur_roi = roi_lat.reshape(-1).to(device)           # [L_q = ls*ls]
        else:
            self._cur_roi = torch.ones(ls * ls, device=device)       # [L_q]

        self._acc.reset()
        self._install()
        try:
            yield
        finally:
            self._remove()
            self._stitch(ty, tx, ref_ry, ref_rx, rth, rtw, lq_h, lq_w)
            self._cur_roi = None

    # ── 글로벌 캔버스 스티칭 ─────────────────────────────────────────────────

    def _stitch(
        self,
        ty: int, tx: int,
        ref_ry: int, ref_rx: int,
        rth: int, rtw: int,
        lq_h: int, lq_w: int,
    ):
        """
        레이어 평균된 [L_ref] 벡터를 Ref 픽셀 공간으로 업샘플하여 캔버스에 누적.

        Shape 흐름:
          [L_ref]  →  [1,1,ls,ls]  (latent grid, ls = tile_size//8)
                   →  [1,1,rth,rtw] (bilinear 업샘플 → ref 타일 픽셀 크기)
                   →  [rth,rtw]    (squeeze)
          canvas[ref_ry:ref_ry+rth, ref_rx:ref_rx+rtw] += map_2d
          cnt   [ref_ry:ref_ry+rth, ref_rx:ref_rx+rtw] += 1.0
        """
        if self._acc.n == 0:
            return

        m1, m2, m3 = self._acc.mean()   # 각각 [L_ref]
        ls = self.latent_sz

        def to_2d(vec: torch.Tensor) -> torch.Tensor:
            # [L_ref] → [1,1,ls,ls] → [1,1,rth,rtw] → [rth,rtw]
            v   = vec.cpu().float().reshape(1, 1, ls, ls)
            mn, mx = v.min(), v.max()
            v   = (v - mn) / (mx - mn + 1e-8)   # 0~1 정규화
            return F.interpolate(
                v, size=(rth, rtw), mode='bilinear', align_corners=False
            ).squeeze()                          # [rth, rtw]

        m1_2d = to_2d(m1)   # [rth, rtw]
        m2_2d = to_2d(m2)
        m3_2d = to_2d(m3)

        # 글로벌 캔버스에 누적 (Overlap 영역은 count 나누기로 평균화)
        r0, r1 = ref_ry, ref_ry + rth
        c0, c1 = ref_rx, ref_rx + rtw
        self.canvas1[r0:r1, c0:c1] += m1_2d
        self.canvas2[r0:r1, c0:c1] += m2_2d
        self.canvas3[r0:r1, c0:c1] += m3_2d
        self.cnt    [r0:r1, c0:c1] += 1.0

    # ── 최종 출력 ─────────────────────────────────────────────────────────────

    def finalize(
        self,
        ref_img:     np.ndarray,   # [ref_h, ref_w, 3]  uint8
        output_path: str,
        colormap:    str = 'jet',
    ):
        """
        3개 글로벌 맵을 ref 이미지와 0.5:0.5 블렌딩.
        [Trust | Verify | Combined] 순 가로 연결 → PNG 저장.

        Shape 흐름:
          canvas / cnt  →  [ref_h, ref_w]   (overlap 평균화)
          blend(hm)     →  [ref_h, ref_w, 3] uint8
          출력           →  [ref_h, ref_w*3, 3] uint8
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cnt = self.cnt.clamp(min=1.0)

        # Overlap 평균화
        g1 = (self.canvas1 / cnt).numpy()   # [ref_h, ref_w]
        g2 = (self.canvas2 / cnt).numpy()
        g3 = (self.canvas3 / cnt).numpy()

        cm_fn = plt.get_cmap(colormap)

        def blend(hm: np.ndarray) -> np.ndarray:
            mn, mx = hm.min(), hm.max()
            hm_n   = (hm - mn) / (mx - mn + 1e-8)                   # 0~1 정규화
            hm_rgb = (cm_fn(hm_n) * 255).astype(np.uint8)[:, :, :3] # [H,W,3]
            return ((ref_img.astype(np.float32) * 0.5 +
                     hm_rgb.astype(np.float32)  * 0.5)
                    .clip(0, 255).astype(np.uint8))                  # [H,W,3]

        b1 = blend(g1)   # Trust
        b2 = blend(g2)   # Verify
        b3 = blend(g3)   # Combined

        # 가로 연결: [Trust | Verify | Combined]
        out = np.concatenate([b1, b2, b3], axis=1)   # [ref_h, ref_w*3, 3]

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(out).save(output_path)
        print(f"[AICGVisualizer] 저장 완료: {output_path}  "
              f"({out.shape[1]}x{out.shape[0]} px)")


# ============================================================================
# 4. CPU 검증 (저사양 환경용 더미 데이터 테스트)
# ============================================================================

if __name__ == "__main__":
    import sys
    print("=" * 62)
    print("AICG 시각화 로직 무결성 검증 (CPU, 더미 데이터)")
    print("논문 수치: L_q=4096, d=1024, M=16, B=1, h=8")
    print("=" * 62)

    torch.manual_seed(42)

    # ── 논문 수치 ──────────────────────────────────────────────
    B        = 1
    L_q      = 4096     # 64×64 latent (512px tile / VAE factor 8)
    L_ref    = 4096
    d        = 1024     # hidden dim
    M        = 16       # learnable summary tokens (논문 M=16)
    h        = 8        # attention heads
    C_head   = d // h   # 128  (per-head dim)
    scale    = 1.0 / math.sqrt(C_head)

    # ── 더미 Q, K (Eq.1) ───────────────────────────────────────
    query = torch.randn(B * h, L_q,   C_head)   # [8, 4096, 128]
    key   = torch.randn(B * h, L_ref, C_head)   # [8, 4096, 128]

    # ── 더미 G (Eq.6) ───────────────────────────────────────────
    gate  = torch.sigmoid(torch.randn(B, L_q))  # [1, 4096]

    # ── 더미 ROI: latent 공간 64×64 중앙 32×32 영역 ────────────
    roi_np = np.zeros((64, 64), dtype=np.float32)
    roi_np[16:48, 16:48] = 1.0
    roi_vec = torch.from_numpy(roi_np.reshape(-1))  # [4096]

    # ── [검증 1] compute_aicg_maps ─────────────────────────────
    print("\n[1] compute_aicg_maps 함수 검증")
    print(f"  query : {list(query.shape)}")    # [8, 4096, 128]
    print(f"  key   : {list(key.shape)}")      # [8, 4096, 128]
    print(f"  gate  : {list(gate.shape)}")     # [1, 4096]
    print(f"  roi   : {list(roi_vec.shape)}")  # [4096]

    m1, m2, m3 = compute_aicg_maps(query, key, gate, roi_vec, scale, B, h)

    print(f"  Map1 Trust    : {list(m1.shape)},  "
          f"min={m1.min():.4f}, max={m1.max():.4f}")   # [4096]
    print(f"  Map2 Verify   : {list(m2.shape)},  "
          f"min={m2.min():.4f}, max={m2.max():.4f}")   # [4096]
    print(f"  Map3 Combined : {list(m3.shape)},  "
          f"min={m3.min():.4f}, max={m3.max():.4f}")   # [4096]
    assert m1.shape == (L_ref,), "Map1 shape 불일치"
    assert m2.shape == (L_ref,), "Map2 shape 불일치"
    assert m3.shape == (L_ref,), "Map3 shape 불일치"

    # ── [검증 2] 글로벌 캔버스 스티칭 ─────────────────────────
    print("\n[2] 글로벌 캔버스 Stitching 검증 (더미 타일 2개, Overlap 포함)")
    ref_h, ref_w = 512, 512
    tile_size    = 512
    ls           = tile_size // 8   # 64

    canvas1 = torch.zeros(ref_h, ref_w)
    canvas2 = torch.zeros(ref_h, ref_w)
    canvas3 = torch.zeros(ref_h, ref_w)
    cnt_map = torch.zeros(ref_h, ref_w)

    def _stitch_dummy(v1, v2, v3, ry, rx, rth, rtw):
        def to_2d(vec):
            v = vec.cpu().float().reshape(1, 1, ls, ls)
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            return F.interpolate(v, size=(rth, rtw), mode='bilinear',
                                 align_corners=False).squeeze()  # [rth, rtw]
        canvas1[ry:ry+rth, rx:rx+rtw] += to_2d(v1)
        canvas2[ry:ry+rth, rx:rx+rtw] += to_2d(v2)
        canvas3[ry:ry+rth, rx:rx+rtw] += to_2d(v3)
        cnt_map[ry:ry+rth, rx:rx+rtw] += 1.0
        print(f"  Stitched tile  ref[{ry}:{ry+rth}, {rx}:{rx+rtw}]  OK")

    # 타일 1: 전체 ref 영역
    _stitch_dummy(m1, m2, m3, 0, 0, 512, 512)
    # 타일 2: 중앙 256×256  (overlap 시뮬레이션)
    _stitch_dummy(m1, m2, m3, 128, 128, 256, 256)

    cnt_clamp = cnt_map.clamp(min=1.0)
    final1 = (canvas1 / cnt_clamp).numpy()   # [512, 512]
    final2 = (canvas2 / cnt_clamp).numpy()
    final3 = (canvas3 / cnt_clamp).numpy()
    print(f"  최종 맵 shape: {final1.shape}  ✓")
    assert final1.shape == (ref_h, ref_w)

    # ── [검증 3] 블렌딩 및 PNG 저장 ───────────────────────────
    print("\n[3] 블렌딩 및 PNG 저장 검증")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ref_dummy = np.random.randint(0, 255, (ref_h, ref_w, 3), dtype=np.uint8)

    def _blend(hm_np, ref_np):
        mn, mx = hm_np.min(), hm_np.max()
        hm_n   = (hm_np - mn) / (mx - mn + 1e-8)
        cm_f   = plt.get_cmap('jet')
        hm_rgb = (cm_f(hm_n) * 255).astype(np.uint8)[:, :, :3]
        return ((ref_np * 0.5 + hm_rgb * 0.5).clip(0, 255)).astype(np.uint8)

    b1 = _blend(final1, ref_dummy)   # [512, 512, 3]
    b2 = _blend(final2, ref_dummy)
    b3 = _blend(final3, ref_dummy)

    # [Trust | Verify | Combined]  →  [512, 1536, 3]
    out_arr  = np.concatenate([b1, b2, b3], axis=1)
    test_png = "./aicg_vis_test.png"
    Image.fromarray(out_arr).save(test_png)
    print(f"  저장: {test_png}  ({out_arr.shape[1]}x{out_arr.shape[0]} px)  ✓")
    assert out_arr.shape == (512, 1536, 3), "출력 shape 불일치"

    # ── 검증 완료 후 임시 파일 삭제 ───────────────────────────
    os.remove(test_png)
    print(f"  임시 파일 삭제: {test_png}  ✓")

    print("\n[검증 완료] 모든 단계가 에러 없이 통과되었습니다.")
