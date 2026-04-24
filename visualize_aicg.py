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
  Map1 Trust    : ROI_mask.T  @ A_RA_avg  [L_ref]
  Map2 Verify   : G.T         @ A_RA_avg  [L_ref]
  Map3 Combined : (ROI * G).T @ A_RA_avg  [L_ref]
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
    query:   torch.Tensor,   # [h, L_q,   C']  단일 타일 (B=1 슬라이스)
    key:     torch.Tensor,   # [h, L_ref,  C']
    gate:    torch.Tensor,   # [1, L_q]         G (Eq.6)
    roi_vec: torch.Tensor,   # [L_q]            ROI 마스크 (0~1)
    scale:   float,          # 1/√C'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A_RA(Eq.2)를 헤드별 계산 후 즉시 [L_ref]로 투영·삭제.
    query.shape[0] 에서 헤드 수를 자동 도출.

    Shape 흐름:
      a_ra_h : [L_q, L_ref]   (헤드 h, 즉시 해제)
      acc*   : [L_ref]        (헤드 평균 누적)
    Returns: (map1, map2, map3)  각각 [L_ref], query.device
    """
    device     = query.device
    num_heads  = query.shape[0]          # 레이어별 실제 헤드 수 자동 도출
    L_ref      = key.shape[1]

    roi   = roi_vec.float().to(device)
    g     = gate[0].float().to(device)  # [L_q]
    roi_g = roi * g

    acc1 = torch.zeros(L_ref, dtype=torch.float32, device=device)
    acc2 = torch.zeros(L_ref, dtype=torch.float32, device=device)
    acc3 = torch.zeros(L_ref, dtype=torch.float32, device=device)

    q32 = query.float()
    k32 = key.float()

    for h_idx in range(num_heads):
        a_ra_h = torch.softmax(q32[h_idx] @ k32[h_idx].T * scale, dim=-1)
        acc1 += roi   @ a_ra_h
        acc2 += g     @ a_ra_h
        acc3 += roi_g @ a_ra_h
        del a_ra_h

    return acc1 / num_heads, acc2 / num_heads, acc3 / num_heads


# ============================================================================
# 2. 타일별 레이어 누적 저장소
# ============================================================================

class _LayerAccumulator:
    """한 타일 forward의 여러 어텐션 레이어 결과를 누적.

    레이어마다 L_ref가 다를 수 있으므로 첫 레이어 크기를 기준으로
    이후 레이어를 1-D linear interpolation 후 누적.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.m1: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None
        self.m3: Optional[torch.Tensor] = None
        self.n:  int = 0

    @staticmethod
    def _resize2d(v: torch.Tensor, target_L: int) -> torch.Tensor:
        """[L] → 2D bilinear resize → [target_L].

        1D linear interpolation crosses row boundaries of the flattened
        spatial map, creating horizontal smearing artifacts.  Reshape to
        (ls, ls) first so the bilinear kernel stays within each spatial
        neighbourhood.
        """
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
            self.m1, self.m2, self.m3 = m1.clone(), m2.clone(), m3.clone()
        else:
            t = self.m1.shape[0]
            self.m1 += self._resize2d(m1, t)
            self.m2 += self._resize2d(m2, t)
            self.m3 += self._resize2d(m3, t)
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

    핵심 설계 (속도 최적화):
      capture_batch() 를 사용하면 기존 tile_batch_size 단위 배치 처리를
      그대로 유지하면서 훅 안에서 배치 차원을 타일별로 분리해 처리함.
      → 시각화 비활성화 시와 동일한 infer_batch 호출 횟수 보장.

    demo_tiled.py 통합 예시:
    ─────────────────────────────────────────────────────────
        viz = AICGVisualizer(net_sr.unet, roi_mask, ref_h, ref_w, tile_size)

        for batch_start in range(0, n_tiles, batch_sz):
            batch_pos   = all_tiles[batch_start : batch_start + batch_sz]
            tile_coords = [...]   # (ty,tx,ry,rx,rth,rtw) per tile

            with viz.capture_batch(tile_coords, lq_h, lq_w):
                predictions = infer_batch(x_src_b, x_ref_b, ...)

        viz.finalize(np.array(ref_img), "output_vis.png")
    ─────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        unet,
        roi_mask:      Optional[np.ndarray],   # [lq_h, lq_w] float32 0~1 or None
        ref_h:         int,
        ref_w:         int,
        tile_size:     int = 512,
        fusion_blocks: str = "full",
    ):
        self.unet      = unet
        self.roi_mask  = roi_mask
        self.ref_h     = ref_h
        self.ref_w     = ref_w
        self.tile_size = tile_size

        self.canvas1 = torch.zeros(ref_h, ref_w)
        self.canvas2 = torch.zeros(ref_h, ref_w)
        self.canvas3 = torch.zeros(ref_h, ref_w)
        self.cnt     = torch.zeros(ref_h, ref_w)

        self._hook_handles:  list                          = []
        self._batch_accs:    Optional[List[_LayerAccumulator]] = None
        self._batch_roi_pix: Optional[List]                = None  # [ts,ts] or None per tile

        self._attn_refs = self._collect_attn_refs(fusion_blocks)
        if not self._attn_refs:
            print("[AICGVisualizer WARNING] attn_ref 모듈을 찾지 못했습니다.")

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
        """배치 단위 훅 설치.

        훅 안에서 배치 차원(B)을 타일별로 분리하여 각자의
        _LayerAccumulator 에 누적 → infer_batch 를 B회 쪼개지 않아도 됨.
        """
        batch_accs   = self._batch_accs    # list[_LayerAccumulator], len=B
        batch_roi    = self._batch_roi_pix  # list[Tensor[ts,ts] or None], len=B
        ts           = self.tile_size

        for attn_ref in self._attn_refs:

            def _make_hook(ar):
                def hook(module, args, kwargs, output):
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
                        # Q [B*h, L_q, C'],  K [B*h, L_ref, C']  – Eq.1
                        q = ar.head_to_batch_dim(ar.to_q(hs))
                        k = ar.head_to_batch_dim(ar.to_k(enc))

                        # G 재현 (Eq.3~6)
                        ts_tok = ar.learnable_token.expand(B_val, -1, -1)
                        ts_k   = ar.head_to_batch_dim(ts_tok)
                        # Eq.4: softmax over L_ref to get per-token weights, then ksum
                        s_ref  = torch.softmax(
                            torch.bmm(ts_k, k.transpose(-1, -2)) * ar.scale, dim=-1)
                        ksum   = torch.bmm(s_ref, k)
                        # Eq.5-6: raw attention scores (no softmax) → batch_to_head_dim
                        # → mean → sigmoid.  Matches attn_processor.py lines 65-67
                        # exactly (processor does NOT softmax attn_summary).
                        smap_raw = torch.bmm(q, ksum.transpose(-1, -2)) * ar.scale
                        gate     = torch.sigmoid(
                            ar.batch_to_head_dim(smap_raw).mean(dim=-1))  # [B, L_q]

                        actual_heads = q.shape[0] // B_val
                        L_q_actual   = q.shape[1]
                        ls_actual    = int(math.sqrt(L_q_actual))

                        # ── 배치 내 타일별 분리 처리 ─────────────────────
                        for i in range(min(B_val, len(batch_accs))):
                            h0, h1 = i * actual_heads, (i + 1) * actual_heads
                            q_i    = q   [h0:h1]     # [h, L_q,   C']
                            k_i    = k   [h0:h1]     # [h, L_ref, C']
                            gate_i = gate[i:i+1]      # [1, L_q]

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
                                q_i, k_i, gate_i, roi_vec, ar.scale
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
        acc: _LayerAccumulator,
        ref_ry: int, ref_rx: int,
        rth: int, rtw: int,
    ):
        """단일 타일의 레이어 평균 맵을 ref 픽셀 공간으로 업샘플해 캔버스에 누적."""
        if acc.n == 0:
            return

        m1, m2, m3 = acc.mean()   # 각각 [L_ref]

        def to_2d(vec: torch.Tensor) -> torch.Tensor:
            ls = int(math.sqrt(vec.shape[0]))   # L_ref에서 역산
            v  = vec.cpu().float().reshape(1, 1, ls, ls)
            mn, mx = v.min(), v.max()
            v  = (v - mn) / (mx - mn + 1e-8)
            return F.interpolate(
                v, size=(rth, rtw), mode='bilinear', align_corners=False
            ).squeeze()   # [rth, rtw]

        r0, r1 = ref_ry, ref_ry + rth
        c0, c1 = ref_rx, ref_rx + rtw
        self.canvas1[r0:r1, c0:c1] += to_2d(m1)
        self.canvas2[r0:r1, c0:c1] += to_2d(m2)
        self.canvas3[r0:r1, c0:c1] += to_2d(m3)
        self.cnt    [r0:r1, c0:c1] += 1.0

    # ── 배치 컨텍스트 매니저 (메인 인터페이스) ───────────────────────────────

    @contextmanager
    def capture_batch(
        self,
        tile_coords: List[Tuple[int, int, int, int, int, int]],
        # [(ty, tx, ref_ry, ref_rx, rth, rtw), ...]
        lq_h: int,
        lq_w: int,
    ):
        """
        배치 내 타일 전체를 한 번의 infer_batch로 처리하면서 시각화.

        기존 tile_batch_size 단위 배치 처리 속도를 유지:
          - infer_batch 호출 횟수: 시각화 미사용과 동일 (배치당 1회)
          - 훅 안에서 배치 차원을 타일별로 분리 → 개별 _LayerAccumulator 에 누적
        """
        ts = self.tile_size
        B  = len(tile_coords)

        # 타일별 픽셀 공간 ROI 준비
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

        # 타일별 레이어 누적기
        self._batch_accs = [_LayerAccumulator() for _ in range(B)]

        self._install()
        try:
            yield
        finally:
            self._remove()
            for i, (ty, tx, ry, rx, rth, rtw) in enumerate(tile_coords):
                self._stitch_one(self._batch_accs[i], ry, rx, rth, rtw)
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
        ref_img:     np.ndarray,
        output_path: str,
        colormap:    str = 'jet',
    ):
        """3개 글로벌 맵을 ref 이미지와 0.5:0.5 블렌딩 → PNG 저장."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cnt   = self.cnt.clamp(min=1.0)
        g1    = (self.canvas1 / cnt).numpy()
        g2    = (self.canvas2 / cnt).numpy()
        g3    = (self.canvas3 / cnt).numpy()
        cm_fn = plt.get_cmap(colormap)

        def blend(hm: np.ndarray) -> np.ndarray:
            mn, mx = hm.min(), hm.max()
            hm_n   = (hm - mn) / (mx - mn + 1e-8)
            hm_rgb = (cm_fn(hm_n) * 255).astype(np.uint8)[:, :, :3]
            return ((ref_img.astype(np.float32) * 0.5 +
                     hm_rgb.astype(np.float32)  * 0.5)
                    .clip(0, 255).astype(np.uint8))

        out = np.concatenate([blend(g1), blend(g2), blend(g3)], axis=1)

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(out).save(output_path)
        print(f"[AICGVisualizer] 저장 완료: {output_path}  "
              f"({out.shape[1]}x{out.shape[0]} px)")


# ============================================================================
# 4. CPU 검증
# ============================================================================

if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print("=" * 60)
    print("AICG 시각화 로직 검증 (CPU, 더미 데이터)")
    print("=" * 60)

    torch.manual_seed(42)

    # ── [1] compute_aicg_maps: 헤드 수 자동 도출 ──────────────
    print("\n[1] compute_aicg_maps – 헤드 수 자동 도출")
    for h, L_q, L_ref in [(8, 4096, 4096), (4, 1024, 1024), (2, 256, 256)]:
        q   = torch.randn(h, L_q,   128)
        k   = torch.randn(h, L_ref, 128)
        g   = torch.sigmoid(torch.randn(1, L_q))
        roi = torch.ones(L_q)
        m1, m2, m3 = compute_aicg_maps(q, k, g, roi, 1/math.sqrt(128))
        assert m1.shape == (L_ref,)
        print(f"  h={h}  L_q={L_q}  L_ref={L_ref}  OK")

    # ── [2] _LayerAccumulator 다해상도 누적 ────────────────────
    print("\n[2] _LayerAccumulator – 다해상도 누적")
    acc = _LayerAccumulator()
    for L in [4096, 1024, 256]:
        acc.update(torch.rand(L), torch.rand(L), torch.rand(L))
    r1, _, _ = acc.mean()
    assert r1.shape == (4096,)
    print(f"  mean shape={r1.shape[0]}  OK")

    # ── [3] capture_batch: 배치 내 타일 분리 ──────────────────
    print("\n[3] capture_batch – 배치 내 타일 분리 시뮬레이션")
    # tile_coords: B=3 타일
    tile_coords = [
        (0,   0,   0,   0,   256, 256),
        (0,   256, 0,   256, 256, 256),
        (256, 0,   256, 0,   256, 256),
    ]
    B      = len(tile_coords)
    h_test = 8
    L_q    = 1024
    L_ref  = 1024
    C_h    = 128
    scale  = 1 / math.sqrt(C_h)

    # 배치 q/k/gate 시뮬레이션
    q_batch = torch.randn(B * h_test, L_q,  C_h)
    k_batch = torch.randn(B * h_test, L_ref, C_h)
    gate_b  = torch.sigmoid(torch.randn(B, L_q))

    batch_accs = [_LayerAccumulator() for _ in range(B)]
    for i in range(B):
        h0, h1 = i * h_test, (i+1) * h_test
        m1, m2, m3 = compute_aicg_maps(
            q_batch[h0:h1], k_batch[h0:h1], gate_b[i:i+1],
            torch.ones(L_q), scale
        )
        batch_accs[i].update(m1, m2, m3)

    # 스티칭 검증
    canvas1 = torch.zeros(512, 512)
    cnt     = torch.zeros(512, 512)
    for i, (ty, tx, ry, rx, rth, rtw) in enumerate(tile_coords):
        m1, _, _ = batch_accs[i].mean()
        ls = int(math.sqrt(m1.shape[0]))
        v  = m1.reshape(1, 1, ls, ls)
        v  = (v - v.min()) / (v.max() - v.min() + 1e-8)
        v2 = F.interpolate(v, size=(rth, rtw), mode='bilinear', align_corners=False).squeeze()
        canvas1[ry:ry+rth, rx:rx+rtw] += v2
        cnt    [ry:ry+rth, rx:rx+rtw] += 1.0
        print(f"  tile {i}: ref[{ry}:{ry+rth},{rx}:{rx+rtw}]  OK")
    final = (canvas1 / cnt.clamp(min=1.0)).numpy()
    assert final.shape == (512, 512)
    print(f"  canvas shape={final.shape}  OK")

    # ── [4] finalize PNG ───────────────────────────────────────
    print("\n[4] finalize PNG 저장")
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref_dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cm = plt.get_cmap('jet')
    def _blend(hm):
        mn, mx = hm.min(), hm.max()
        n = (hm - mn)/(mx - mn + 1e-8)
        return ((ref_dummy*0.5 + (cm(n)*255).astype(np.uint8)[:,:,:3]*0.5)
                .clip(0,255).astype(np.uint8))
    out = np.concatenate([_blend(final[:256,:256])]*3, axis=1)
    assert out.shape == (256, 768, 3)
    test_png = "./aicg_vis_test.png"
    Image.fromarray(out).save(test_png)
    os.remove(test_png)
    print(f"  PNG 저장·삭제  OK")

    print("\n[검증 완료] 모든 단계 통과.")
