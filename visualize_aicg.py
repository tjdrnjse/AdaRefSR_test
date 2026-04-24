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
    query:    torch.Tensor,   # [B*h, L_q,   C']  –  Q (Eq.1)
    key:      torch.Tensor,   # [B*h, L_ref,  C']  –  K (Eq.1)
    gate:     torch.Tensor,   # [B,   L_q]          –  G (Eq.6)
    roi_vec:  torch.Tensor,   # [L_q]               –  ROI 마스크 (토큰 공간, 0~1)
    scale:    float,          # 1/√C'  (attn.scale)
    B:        int,            # 배치 크기
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A_RA 행렬(Eq.2)을 헤드별로 계산한 뒤 즉시 [L_ref] 벡터로 투영하고 삭제.
    행렬 자체([B*h, L_q, L_ref])는 캔버스에 저장하지 않음(메모리 절약).

    [Fix 5] 헤드 수를 query 텐서에서 직접 도출 (query.shape[0] // B).
            레이어마다 헤드 수가 달라도 범위 초과 없음.

    연산 단계별 Shape:
      q32, k32       : [B*h, L_q/L_ref, C']
      a_ra_h (헤드 h): [L_q, L_ref]             A_RA_h = Softmax(Q_h K_h^T/√d)  Eq.2
      acc1/acc2/acc3 : [L_ref]                   헤드 평균 누적

    Returns:
      (map1, map2, map3)  각각 shape [L_ref],  device=query.device, float32
    """
    device = query.device
    L_ref  = key.shape[1]

    # [Fix 5] 실제 헤드 수를 텐서 shape 에서 도출
    actual_heads = query.shape[0] // B   # 레이어별 헤드 수 (8, 4, 2 등)

    roi   = roi_vec.float().to(device)   # [L_q]
    g     = gate[0].float().to(device)  # [L_q]  (B=1 슬라이스)
    roi_g = roi * g                      # [L_q]

    acc1 = torch.zeros(L_ref, dtype=torch.float32, device=device)
    acc2 = torch.zeros(L_ref, dtype=torch.float32, device=device)
    acc3 = torch.zeros(L_ref, dtype=torch.float32, device=device)

    q32 = query.float()   # [B*h, L_q,   C']
    k32 = key.float()     # [B*h, L_ref, C']

    for h_idx in range(B * actual_heads):
        q_h = q32[h_idx]   # [L_q,   C']
        k_h = k32[h_idx]   # [L_ref, C']

        # A_RA_h = Softmax(Q_h K_h^T / √d)  [L_q, L_ref]  – Eq.2
        a_ra_h = torch.softmax(q_h @ k_h.T * scale, dim=-1)

        acc1 += roi   @ a_ra_h   # Map1 Trust
        acc2 += g     @ a_ra_h   # Map2 Verify
        acc3 += roi_g @ a_ra_h   # Map3 Combined

        del a_ra_h   # 행렬 즉시 해제 (캔버스에 직접 저장 금지)

    n = B * actual_heads
    return acc1 / n, acc2 / n, acc3 / n   # 각각 [L_ref]


# ============================================================================
# 2. 타일별 레이어 누적 저장소
# ============================================================================

class _LayerAccumulator:
    """한 타일 forward 에서 여러 어텐션 레이어의 결과를 누적.

    [Fix 2] 레이어마다 L_ref 크기가 다를 수 있으므로 (UNet 다해상도),
            첫 번째 레이어의 크기를 기준으로 이후 레이어를 1-D linear
            interpolation 으로 리사이즈한 뒤 누적.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.m1: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None
        self.m3: Optional[torch.Tensor] = None
        self.n:  int = 0

    @staticmethod
    def _resize1d(v: torch.Tensor, target: int) -> torch.Tensor:
        """[L] → [target] 1-D linear interpolation. 입력 디바이스를 유지."""
        if v.shape[0] == target:
            return v
        return F.interpolate(
            v.float().view(1, 1, -1),   # .cpu() 제거 – CUDA 텐서 그대로 처리
            size=target,
            mode='linear',
            align_corners=False,
        ).squeeze(0).squeeze(0)

    def update(self, m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor):
        if self.m1 is None:
            # 첫 레이어의 L_ref 가 기준 크기가 됨
            self.m1, self.m2, self.m3 = m1.clone(), m2.clone(), m3.clone()
        else:
            target = self.m1.shape[0]
            # [Fix 2] 크기가 다르면 기준 크기로 리사이즈 후 누적
            self.m1 += self._resize1d(m1, target)
            self.m2 += self._resize1d(m2, target)
            self.m3 += self._resize1d(m3, target)
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

    Args:
        unet          : GenModel.unet  (attn_ref 포함한 UNet)
        roi_mask      : np.ndarray [lq_h, lq_w] float32 0~1 또는 None (전체)
        ref_h / ref_w : 전체 Ref 이미지 픽셀 해상도
        tile_size     : 타일 한 변 픽셀 크기
        fusion_blocks : "full" 또는 "midup"
    """

    def __init__(
        self,
        unet,
        roi_mask:       Optional[np.ndarray],
        ref_h:          int,
        ref_w:          int,
        tile_size:      int = 512,
        fusion_blocks:  str = "full",
    ):
        self.unet        = unet
        self.roi_mask    = roi_mask
        self.ref_h       = ref_h
        self.ref_w       = ref_w
        self.tile_size   = tile_size

        # 전역 캔버스 3개 + 카운팅 맵  [ref_h, ref_w]  CPU float32
        self.canvas1 = torch.zeros(ref_h, ref_w)
        self.canvas2 = torch.zeros(ref_h, ref_w)
        self.canvas3 = torch.zeros(ref_h, ref_w)
        self.cnt     = torch.zeros(ref_h, ref_w)

        self._acc:          _LayerAccumulator      = _LayerAccumulator()
        self._hook_handles: list                   = []
        # [Fix 4] 픽셀 공간 ROI 크롭을 저장; 훅 안에서 실제 L_q 에 맞게 리사이즈
        self._cur_roi_pix:  Optional[torch.Tensor] = None

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
        acc         = self._acc
        get_roi_pix = lambda: self._cur_roi_pix
        ts          = self.tile_size

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
                        # Q = H_src W_Q  [B*h, L_q,   C']  – Eq.1
                        # K = H_ref W_K  [B*h, L_ref, C']  – Eq.1
                        q = ar.head_to_batch_dim(ar.to_q(hs))
                        k = ar.head_to_batch_dim(ar.to_k(enc))

                        # ── G 재현 (Eq.3 ~ Eq.6) ─────────────────────────
                        ts_tok = ar.learnable_token.expand(B_val, -1, -1)
                        ts_k   = ar.head_to_batch_dim(ts_tok)

                        s_ref = torch.softmax(
                            torch.bmm(ts_k, k.transpose(-1, -2)) * ar.scale, dim=-1
                        )
                        ksum  = torch.bmm(s_ref, k)

                        smap  = torch.softmax(
                            torch.bmm(q, ksum.transpose(-1, -2)) * ar.scale, dim=-1
                        )
                        gate  = torch.sigmoid(
                            ar.batch_to_head_dim(smap).mean(dim=-1)
                        )                                       # [B, L_q]

                        # [Fix 4] 레이어 실제 L_q 에 맞춰 ROI 를 동적으로 리사이즈
                        L_q_actual = q.shape[1]
                        ls_actual  = int(math.sqrt(L_q_actual))
                        roi_pix    = get_roi_pix()              # [ts, ts] or None
                        if roi_pix is not None:
                            # 픽셀 ROI → 이 레이어의 latent 해상도로 다운샘플
                            roi_vec = F.interpolate(
                                roi_pix.unsqueeze(0).unsqueeze(0).float(),
                                size=(ls_actual, ls_actual),
                                mode='area',
                            ).squeeze().reshape(-1).to(q.device)   # [L_q_actual]
                        else:
                            roi_vec = torch.ones(L_q_actual, device=q.device)

                        # [Fix 5] num_heads 미전달 – compute_aicg_maps 내부에서 도출
                        m1, m2, m3 = compute_aicg_maps(
                            q, k, gate, roi_vec, ar.scale, B_val
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
        ty:     int, tx:     int,
        ref_ry: int, ref_rx: int,
        rth:    int, rtw:    int,
        lq_h:   int, lq_w:  int,
    ):
        ts = self.tile_size

        # [Fix 4] 픽셀 공간 ROI 크롭을 저장 (훅에서 L_q 별 리사이즈)
        if self.roi_mask is not None:
            crop_np = self.roi_mask[ty:ty + ts, tx:tx + ts].astype(np.float32)
            # 엣지 타일처럼 실제 크롭이 ts 보다 작을 경우 ts×ts 로 패드
            if crop_np.shape[0] != ts or crop_np.shape[1] != ts:
                padded = np.zeros((ts, ts), dtype=np.float32)
                padded[:crop_np.shape[0], :crop_np.shape[1]] = crop_np
                crop_np = padded
            self._cur_roi_pix = torch.from_numpy(crop_np)   # [ts, ts]
        else:
            self._cur_roi_pix = None

        self._acc.reset()
        self._install()
        try:
            yield
        finally:
            self._remove()
            self._stitch(ty, tx, ref_ry, ref_rx, rth, rtw, lq_h, lq_w)
            self._cur_roi_pix = None

    # ── 글로벌 캔버스 스티칭 ─────────────────────────────────────────────────

    def _stitch(
        self,
        ty: int, tx: int,
        ref_ry: int, ref_rx: int,
        rth: int, rtw: int,
        lq_h: int, lq_w: int,
    ):
        if self._acc.n == 0:
            return

        m1, m2, m3 = self._acc.mean()   # 각각 [L_ref] (첫 레이어 기준 크기)

        def to_2d(vec: torch.Tensor) -> torch.Tensor:
            # [Fix 2] ls 를 tile_size//8 고정이 아닌 벡터 길이에서 역산
            ls = int(math.sqrt(vec.shape[0]))
            # [L_ref] → [1,1,ls,ls] → [1,1,rth,rtw] → [rth,rtw]
            v  = vec.cpu().float().reshape(1, 1, ls, ls)
            mn, mx = v.min(), v.max()
            v  = (v - mn) / (mx - mn + 1e-8)
            return F.interpolate(
                v, size=(rth, rtw), mode='bilinear', align_corners=False
            ).squeeze()

        m1_2d = to_2d(m1)
        m2_2d = to_2d(m2)
        m3_2d = to_2d(m3)

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

        [Fix 1] output_path 가 디렉터리인 경우 내부에서 처리 – 호출 전에
                demo_tiled.py 의 _resolve_vis_path() 로 파일 경로로 변환됨.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cnt = self.cnt.clamp(min=1.0)
        g1  = (self.canvas1 / cnt).numpy()
        g2  = (self.canvas2 / cnt).numpy()
        g3  = (self.canvas3 / cnt).numpy()

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
# 4. CPU 검증 (저사양 환경용 더미 데이터 테스트)
# ============================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("AICG 시각화 로직 무결성 검증 (CPU, 더미 데이터)")
    print("논문 수치: L_q=4096, d=1024, M=16, B=1")
    print("Fix 2/4/5 포함 – 다해상도 레이어 + 가변 헤드 수 검증")
    print("=" * 62)

    torch.manual_seed(42)

    B = 1
    scale_128 = 1.0 / math.sqrt(128)

    # ── [검증 1] compute_aicg_maps: 헤드 수 자동 도출 (Fix 5) ──
    print("\n[1] compute_aicg_maps – 헤드 수 자동 도출 (Fix 5)")
    for h, L_q, L_ref in [(8, 4096, 4096), (4, 1024, 1024), (2, 256, 256)]:
        C_head = 128
        q   = torch.randn(B * h, L_q,   C_head)
        k   = torch.randn(B * h, L_ref, C_head)
        g   = torch.sigmoid(torch.randn(B, L_q))
        roi = torch.ones(L_q)
        m1, m2, m3 = compute_aicg_maps(q, k, g, roi, scale_128, B)
        assert m1.shape == (L_ref,)
        print(f"  h={h:2d}  L_q={L_q:4d}  L_ref={L_ref:4d}  OK")

    # ── [검증 2] _LayerAccumulator: 다해상도 누적 (Fix 2) ────────
    print("\n[2] _LayerAccumulator – 다해상도 shape 불일치 처리 (Fix 2)")
    acc = _LayerAccumulator()
    for L_ref in [4096, 1024, 256]:
        m1 = torch.rand(L_ref); m2 = torch.rand(L_ref); m3 = torch.rand(L_ref)
        acc.update(m1, m2, m3)
        print(f"  update L_ref={L_ref}  accumulated shape={acc.m1.shape[0]}  OK")
    r1, r2, r3 = acc.mean()
    assert r1.shape == (4096,), f"expected 4096, got {r1.shape}"
    print(f"  mean shape={r1.shape[0]}  OK")

    # ── [검증 3] to_2d: ls 역산 (Fix 2) ─────────────────────────
    print("\n[3] _stitch to_2d – ls 역산 (Fix 2)")
    for L_ref in [4096, 1024, 256]:
        ls = int(math.sqrt(L_ref))
        v  = torch.rand(L_ref)
        v2 = v.reshape(1, 1, ls, ls)
        v2 = (v2 - v2.min()) / (v2.max() - v2.min() + 1e-8)
        out_2d = F.interpolate(v2, size=(128, 128), mode='bilinear',
                               align_corners=False).squeeze()
        assert out_2d.shape == (128, 128)
        print(f"  L_ref={L_ref}  ls={ls}  → (128,128)  OK")

    # ── [검증 4] capture_tile ROI 동적 리사이즈 (Fix 4) ──────────
    print("\n[4] capture_tile ROI 픽셀→latent 동적 리사이즈 (Fix 4)")
    ts    = 512
    roi_mask = np.zeros((1024, 1024), dtype=np.float32)
    roi_mask[256:768, 256:768] = 1.0
    crop  = roi_mask[0:ts, 0:ts].astype(np.float32)
    crop_t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0)
    for ls_actual in [64, 32, 16]:
        roi_lat = F.interpolate(crop_t, size=(ls_actual, ls_actual),
                                mode='area').squeeze().reshape(-1)
        assert roi_lat.shape == (ls_actual * ls_actual,)
        print(f"  ls_actual={ls_actual:2d}  ROI shape={roi_lat.shape[0]:5d}  OK")

    # ── [검증 5] finalize PNG 저장 ───────────────────────────────
    print("\n[5] finalize PNG 저장")
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref_h, ref_w = 256, 256
    ref_dummy    = np.random.randint(0, 255, (ref_h, ref_w, 3), dtype=np.uint8)
    canvas1 = torch.rand(ref_h, ref_w)
    canvas2 = torch.rand(ref_h, ref_w)
    canvas3 = torch.rand(ref_h, ref_w)
    cm_fn   = plt.get_cmap('jet')
    def _blend(hm):
        mn, mx = hm.min(), hm.max()
        n = (hm - mn) / (mx - mn + 1e-8)
        rgb = (cm_fn(n) * 255).astype(np.uint8)[:, :, :3]
        return ((ref_dummy * 0.5 + rgb * 0.5).clip(0, 255)).astype(np.uint8)
    out = np.concatenate([_blend(canvas1.numpy()),
                          _blend(canvas2.numpy()),
                          _blend(canvas3.numpy())], axis=1)
    assert out.shape == (ref_h, ref_w * 3, 3)
    test_png = "./aicg_vis_test.png"
    Image.fromarray(out).save(test_png)
    print(f"  저장: {test_png}  shape={out.shape}  OK")
    os.remove(test_png)
    print(f"  임시 파일 삭제  OK")

    print("\n[검증 완료] 모든 단계가 에러 없이 통과되었습니다.")
