"""
aicg_steering.py – AICG Trust/Verify intervention via processor-level hooks
============================================================================

Ada-RefSR 의 reference attention (`attn_ref`) 모듈에 직접 개입(Intervention)하여
훼손된 LR 얼굴 영역에서 Ref 의존성을 강제로 끌어올리는 AICGSteerer 를 제공.

AICG 모듈 내부 연산 흐름 (paper Eq.1~7):
    Q = to_q(H_src),   K = to_k(H_ref),   V = to_v(H_ref)
    A_RA = Softmax(QK^T / √d)                              [Trust attention]
    Ksum = Softmax(T·K^T / √d) @ K                          [reference summary]
    Smap = Softmax(Q·Ksum^T / √d)
    G    = σ( mean_M,heads(Smap) )                          [Verify gate]
    Out  = ZeroLinear( A_RA @ V ) ⊙ G + H_src

Intervention:
    Trust 강화 (Logit Scaling):
        face_mask 영역의 query 위치에서 Softmax 직전 logit 에 trust_scale 곱
        → A_RA 가 Ref 토큰에 더 sharp 하게 집중
    Verify 강화 (Gate G Forcing):
        face_mask 영역에서 G ← clamp(G * verify_scale, 0, 1)
        → AICG 가 Ref 정보를 더 강하게 통과시킴

구현 메커니즘:
    diffusers 의 attn_ref.processor 를 SteeredReferenceAttnProcessor 로 교체.
    Steerer 의 컨텍스트(`apply_steering()`) 진입 시 교체, 종료 시 원복.

    원본 attn_processor_valid_high 와 수치적으로 동등한 path (face_mask=None 또는
    scale=1.0) 를 fall-through 로 유지하여 비활성화 모드에서 회귀 위험 0.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _torch_dfs(module: nn.Module):
    out = [module]
    for child in module.children():
        out += _torch_dfs(child)
    return out


# ============================================================================
# Steered processor – mirrors attn_processor_valid_high but with intervention
# ============================================================================

class SteeredReferenceAttnProcessor:
    """
    drop-in replacement for ReferenceAttnProcessorWithZeroConvolution.

    Trust intervention: pre-softmax logit scaling on face_mask query rows.
    Verify intervention: post-sigmoid gate G clamped scaling on face_mask rows.

    when steerer.is_active() == False  OR  face_mask is None  OR
    both scales equal 1.0 → SDPA fast-path is taken (identical to the
    unmodified inference processor).
    """

    def __init__(self, steerer: "AICGSteerer"):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch >= 2.0 for scaled_dot_product_attention.")
        self.steerer = steerer
        # Per-layer visualization cache (overwritten each forward, read by hook synchronously)
        # _last_probs : [B*h, L_q, L_ref]  Trust attention map (post-intervention)
        # _last_gate  : [B,   L_q]          Verify gate         (post-intervention)
        self._last_probs: Optional[torch.Tensor] = None
        self._last_gate:  Optional[torch.Tensor] = None

    # ────────────────────────────────────────────────────────────────────────

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        external_kv=None,
        temb=None,
    ):
        # ── Shape normalization ────────────────────────────────────────────
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, -1).transpose(1, 2)

        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)
        if encoder_hidden_states.ndim == 4:
            ek_b, ek_c, ek_h, ek_w = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(ek_b, ek_c, -1).transpose(1, 2)

        B = hidden_states.shape[0]

        # ── Q, K, V projections (unchanged) ────────────────────────────────
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))            # [B*h, L_q, C']
        key   = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))    # [B*h, K,   C']
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))    # [B*h, K,   C']

        L_q   = query.shape[1]
        h_eff = query.shape[0] // B    # 실제 head 수 (heads // chunked dim)

        # ── Reference summary (Eq.3~4) ─────────────────────────────────────
        learnable_token = attn.learnable_token.expand(B, -1, -1)
        learnable_token = attn.head_to_batch_dim(learnable_token)
        attn_ref_logits = torch.bmm(learnable_token, key.transpose(-1, -2)) * attn.scale
        attn_ref_w      = torch.softmax(attn_ref_logits, dim=-1)
        summarized_token = torch.bmm(attn_ref_w, key)

        # ── Verify gate G (Eq.5~6) ─────────────────────────────────────────
        attn_summary = torch.bmm(query, summarized_token.transpose(-1, -2)) * attn.scale
        attn_summary = attn.batch_to_head_dim(attn_summary)         # [B, L_q, M*h]
        gate         = torch.sigmoid(attn_summary.mean(dim=-1))     # [B, L_q]

        # ── Build per-token face mask if steering is active ────────────────
        steer = self.steerer.is_active()
        face_mask_q_2d: Optional[torch.Tensor] = None   # [B, L_q] bool
        face_mask_q_bh: Optional[torch.Tensor] = None   # [B*h, L_q] bool (for trust)

        if steer:
            face_mask_q_2d = self.steerer.build_query_mask(
                B=B, L_q=L_q, device=query.device,
            )
            if face_mask_q_2d is not None:
                # broadcast across heads → [B*h, L_q]
                face_mask_q_bh = face_mask_q_2d.repeat_interleave(h_eff, dim=0)

        # ── Verify intervention: face_mask 영역에 한해 gate 값을 조정 ─────────
        #   조건 A  force_verify=True  → 무조건 1.0 (Hard Override, verify_scale 무관)
        #   조건 B  force_verify=False + effective_verify_scale != 1.0
        #                              → clamp(gate * verify_scale, 0, 1)  (기존 방식)
        if steer and face_mask_q_2d is not None:
            if self.steerer.force_verify:
                forced = torch.ones_like(gate)                                # Hard Override
            elif self.steerer.effective_verify_scale != 1.0:
                forced = (gate * self.steerer.effective_verify_scale).clamp(0.0, 1.0)
            else:
                forced = None
            if forced is not None:
                gate = torch.where(face_mask_q_2d, forced, gate)

        # ── Main attention (Eq.2)  – Trust intervention applied if needed ──
        chunk_size = 1024
        output     = torch.zeros_like(query)

        trust_active = (
            steer and face_mask_q_bh is not None
            and self.steerer.effective_trust_scale != 1.0
        )

        _viz = self.steerer.capture_for_viz

        if trust_active:
            # 명시적 softmax 경로 (logit 스케일링 가능)
            ts = self.steerer.effective_trust_scale
            _probs_chunks: List[torch.Tensor] = []
            for i in range(0, L_q, chunk_size):
                q_chunk  = query[:, i:i + chunk_size, :]                       # [B*h, c, C']
                scores   = torch.bmm(q_chunk, key.transpose(-1, -2)) * attn.scale
                m_chunk  = face_mask_q_bh[:, i:i + chunk_size].unsqueeze(-1)   # [B*h, c, 1]
                scores   = torch.where(m_chunk, scores * ts, scores)
                probs_c  = torch.softmax(scores, dim=-1)                        # [B*h, c, L_ref]
                out_c    = torch.bmm(probs_c, value)
                output[:, i:i + chunk_size, :] = out_c
                if _viz:
                    _probs_chunks.append(probs_c.detach().cpu())
            # shape: [B*h, L_q, L_ref]  Trust attention map (post-trust-intervention)
            if _viz and _probs_chunks:
                self.steerer._last_probs = torch.cat(_probs_chunks, dim=1)
        else:
            # SDPA fast-path (수치적으로 원본과 동일)
            _sdpa_probs: List[torch.Tensor] = []
            for i in range(0, L_q, chunk_size):
                q_chunk = query[:, i:i + chunk_size, :]
                out_c   = F.scaled_dot_product_attention(
                    q_chunk, key, value,
                    dropout_p=0.0,
                    is_causal=False,
                )
                output[:, i:i + chunk_size, :] = out_c
                if _viz:
                    # SDPA는 probs를 노출하지 않으므로 별도로 계산 (viz 전용, forward 결과와 무관)
                    with torch.no_grad():
                        s_c = torch.bmm(
                            q_chunk.float(), key.float().transpose(-1, -2)
                        ) * attn.scale                                          # [B*h, c, L_ref]
                        _sdpa_probs.append(torch.softmax(s_c, dim=-1).cpu())
            # shape: [B*h, L_q, L_ref]  Trust attention map (no trust intervention → unmodified)
            if _viz and _sdpa_probs:
                self.steerer._last_probs = torch.cat(_sdpa_probs, dim=1)

        # Verify-intervened gate: [B, L_q]  (post-verify-intervention)
        if _viz:
            self.steerer._last_gate = gate.detach().cpu()

        weighted_values = output

        # ── Project back ────────────────────────────────────────────────────
        hidden_states_out = attn.batch_to_head_dim(weighted_values)
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)

        hidden_states_out = attn.zero_linear(hidden_states_out)
        hidden_states_out = hidden_states_out * gate.unsqueeze(-1)

        if input_ndim == 4:
            hidden_states_out = hidden_states_out.transpose(1, 2).view(
                batch_size, channel, height, width
            )
        return hidden_states_out


# ============================================================================
# AICGSteerer – context-managed processor swap
# ============================================================================

class AICGSteerer:
    """
    Trust/Verify intervention 컨트롤러.

    사용 패턴 (demo_tiled.py 통합):
        steerer = AICGSteerer(
            net_sr.unet,
            scale=1.0,             # 전역 강화 배율 (trust/verify 양쪽에 곱)
            trust_scale=1.8,       # 개별 trust 배율 (Q·K^T logit pre-softmax)
            verify_scale=2.0,      # 개별 verify 배율 (Gate G post-sigmoid)
            fusion_blocks="full",
        )

        for batch_start in range(0, n_tiles, batch_sz):
            tile_face_masks = [...]    # [Tensor[H,W] or None] per tile
            steerer.set_batch_face_masks(tile_face_masks)
            with steerer.apply_steering():
                predictions = infer_batch(...)
            steerer.clear_batch_face_masks()

    개별 효과 비활성화:
        scale=1.0, trust_scale=1.0, verify_scale=1.0  → 완전한 no-op
        face_mask 가 모두 None      → 자동 no-op (SDPA fast-path)
    """

    def __init__(
        self,
        unet,
        scale:         float = 1.0,
        trust_scale:   float = 1.0,
        verify_scale:  float = 1.0,
        force_verify:  bool  = False,
        fusion_blocks: str   = "full",
        verbose:       bool  = False,
    ):
        self.unet          = unet
        self.scale         = float(scale)
        self.trust_scale   = float(trust_scale)
        self.verify_scale  = float(verify_scale)
        # force_verify=True: face_mask 영역 gate 를 verify_scale 무관하게 무조건 1.0 으로 강제
        self.force_verify  = bool(force_verify)
        self.fusion_blocks = fusion_blocks
        self.verbose       = verbose

        # per-batch (re-set every tile-batch)
        self._batch_face_masks: Optional[List[Optional[torch.Tensor]]] = None

        # visualization cache: set True to make SteeredReferenceAttnProcessor
        # cache probs/gate each layer forward (read by AICGVisualizer hook)
        self.capture_for_viz: bool = False
        # Layer-level cache (overwritten per-layer, read synchronously by hook)
        # _last_probs : [B*h, L_q, L_ref]  Trust attention map (post-intervention), cpu
        # _last_gate  : [B,   L_q]          Verify gate         (post-intervention), cpu
        self._last_probs: Optional[torch.Tensor] = None
        self._last_gate:  Optional[torch.Tensor] = None

        # active during apply_steering() context
        self._active                  = False
        self._original_processors:    list = []
        self._steered_processors:     list = []
        self._attn_ref_modules:       list = []

        self._collect_attn_ref_modules()

    # ── effective scales ────────────────────────────────────────────────────

    @property
    def effective_trust_scale(self) -> float:
        return self.scale * self.trust_scale

    @property
    def effective_verify_scale(self) -> float:
        return self.scale * self.verify_scale

    def is_active(self) -> bool:
        return self._active

    # ── module collection ───────────────────────────────────────────────────

    def _collect_attn_ref_modules(self):
        if self.unet is None:
            return
        try:
            from diffusers.models.attention import BasicTransformerBlock
        except ImportError:
            return

        if self.fusion_blocks == "midup":
            cands = _torch_dfs(self.unet.mid_block) + _torch_dfs(self.unet.up_blocks)
        else:
            cands = _torch_dfs(self.unet)

        self._attn_ref_modules = [
            m.attn_ref for m in cands
            if isinstance(m, BasicTransformerBlock) and hasattr(m, "attn_ref")
        ]
        if self.verbose:
            print(f"[AICGSteerer] attn_ref modules collected: "
                  f"{len(self._attn_ref_modules)}")

    # ── per-batch face mask interface ───────────────────────────────────────

    def set_batch_face_masks(self, masks: List[Optional[torch.Tensor]]):
        """매 tile-batch 마다 호출. 길이=B, 각 원소는 픽셀 공간 [H, W] mask 또는 None.

        값 범위: 0/1 권장 (soft mask 도 가능). 0.5 임계로 token-level bool 화.
        """
        self._batch_face_masks = list(masks)

    def clear_batch_face_masks(self):
        self._batch_face_masks = None

    def build_query_mask(
        self,
        B:      int,
        L_q:    int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """현재 batch face mask 들을 attention token 해상도(L_q) 로 다운샘플 → [B, L_q] bool.

        모든 tile 의 face mask 가 None 이면 None 반환 → no-op.
        """
        if not self._batch_face_masks:
            return None
        if all(m is None for m in self._batch_face_masks):
            return None

        ls = int(math.sqrt(L_q))
        if ls * ls != L_q:
            # 비정사각 attention map (드물게 발생) → fallback: face mask 비활성
            return None

        # B 가 face_mask 길이보다 클 수 있음(CFG 등). 부족분은 None 처리.
        out = torch.zeros(B, L_q, dtype=torch.bool, device=device)
        for i in range(min(B, len(self._batch_face_masks))):
            m = self._batch_face_masks[i]
            if m is None:
                continue
            with torch.no_grad():
                m_lr = F.interpolate(
                    m.float().unsqueeze(0).unsqueeze(0).to(device=device),
                    size=(ls, ls),
                    mode='area',
                ).squeeze().reshape(-1)              # [L_q]
            out[i] = m_lr > 0.5
        return out

    # ── processor swap context manager ──────────────────────────────────────

    @contextmanager
    def apply_steering(self):
        """attn_ref 모듈 의 processor 를 SteeredReferenceAttnProcessor 로 교체.

        진입 시: 원본 processor 저장 → 교체.
        종료 시: 원본 복원 (예외 발생 시에도 보장).
        """
        if not self._attn_ref_modules:
            # 모듈을 찾지 못했으면 그냥 통과
            self._active = True
            try:
                yield self
            finally:
                self._active = False
            return

        self._original_processors = []
        self._steered_processors  = []
        for ar in self._attn_ref_modules:
            self._original_processors.append(ar.processor)
            steered = SteeredReferenceAttnProcessor(self)
            self._steered_processors.append(steered)
            ar.set_processor(steered)

        if self.verbose:
            print(f"[AICGSteerer] steering ON  "
                  f"(scale={self.scale}, trust={self.trust_scale}, verify={self.verify_scale}, "
                  f"effective_trust={self.effective_trust_scale}, "
                  f"effective_verify={self.effective_verify_scale})")

        self._active = True
        try:
            yield self
        finally:
            for ar, orig in zip(self._attn_ref_modules, self._original_processors):
                ar.set_processor(orig)
            self._active = False
            self._original_processors = []
            self._steered_processors  = []
            if self.verbose:
                print("[AICGSteerer] steering OFF (processors restored)")
