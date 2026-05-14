"""
aicg_steering.py – AICG Trust/Verify intervention via processor-level hooks
============================================================================

Ada-RefSR reference attention ('attn_ref') 모듈에 직접 개입(Intervention)하여
Ref 의존성을 강제로 끌어올리는 AICGSteerer 를 제공.

AICG 모듈 내부 연산 흐름 (paper Eq.1~7):
    Q = to_q(H_src),   K = to_k(H_ref),   V = to_v(H_ref)
    A_RA = Softmax(QK^T / √d)
    Ksum = Softmax(T·K^T / √d) @ K
    Smap = Softmax(Q·Ksum^T / √d)
    G    = σ( mean_M,heads(Smap) )
    Out  = ZeroLinear( A_RA @ V ) ⊙ G + H_src

Intervention (전체 토큰에 적용):
    Trust 강화 (Logit Scaling):
        모든 query 위치에서 Softmax 직전 logit 에 trust_scale 곱
        → A_RA 가 Ref 토큰에 더 sharp 하게 집중
    Verify 강화 (Gate G Forcing):
        모든 위치에서 G ← clamp(G * verify_scale, 0, 1)
        → AICG 가 Ref 정보를 더 강하게 통과시킴

구현 메커니즘:
    diffusers 의 attn_ref.processor 를 SteeredReferenceAttnProcessor 로 교체.
    Steerer 의 컨텍스트(`apply_steering()`) 진입 시 교체, 종료 시 원복.
"""
from __future__ import annotations

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

    Trust intervention: pre-softmax logit scaling on all query tokens.
    Verify intervention: post-sigmoid gate G scaling on all tokens.

    when steerer.is_active() == False OR both scales equal 1.0 → SDPA fast-path.
    """

    def __init__(self, steerer: "AICGSteerer"):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch >= 2.0 for scaled_dot_product_attention.")
        self.steerer = steerer
        self._last_probs: Optional[torch.Tensor] = None
        self._last_gate:  Optional[torch.Tensor] = None

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

        # ── Q, K, V projections ────────────────────────────────────────────
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key   = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))

        L_q = query.shape[1]

        # ── Reference summary (Eq.3~4) ─────────────────────────────────────
        learnable_token  = attn.learnable_token.expand(B, -1, -1)
        learnable_token  = attn.head_to_batch_dim(learnable_token)
        attn_ref_logits  = torch.bmm(learnable_token, key.transpose(-1, -2)) * attn.scale
        attn_ref_w       = torch.softmax(attn_ref_logits, dim=-1)
        summarized_token = torch.bmm(attn_ref_w, key)

        # ── Verify gate G (Eq.5~6) ─────────────────────────────────────────
        attn_summary = torch.bmm(query, summarized_token.transpose(-1, -2)) * attn.scale
        attn_summary = attn.batch_to_head_dim(attn_summary)
        gate         = torch.sigmoid(attn_summary.mean(dim=-1))  # [B, L_q]

        steer = self.steerer.is_active()

        # ── Verify intervention: apply to ALL gate positions ───────────────
        if steer:
            if self.steerer.force_verify:
                gate = torch.ones_like(gate)
            elif self.steerer.effective_verify_scale != 1.0:
                gate = (gate * self.steerer.effective_verify_scale).clamp(0.0, 1.0)

        # ── Main attention (Eq.2) – Trust intervention applied if needed ───
        chunk_size   = 1024
        output       = torch.zeros_like(query)
        trust_active = steer and self.steerer.effective_trust_scale != 1.0
        _viz         = self.steerer.capture_for_viz

        if trust_active:
            ts = self.steerer.effective_trust_scale
            _probs_chunks: List[torch.Tensor] = []
            for i in range(0, L_q, chunk_size):
                q_chunk = query[:, i:i + chunk_size, :]
                scores  = torch.bmm(q_chunk, key.transpose(-1, -2)) * attn.scale * ts
                probs_c = torch.softmax(scores, dim=-1)
                out_c   = torch.bmm(probs_c, value)
                output[:, i:i + chunk_size, :] = out_c
                if _viz:
                    _probs_chunks.append(probs_c.detach().cpu())
            if _viz and _probs_chunks:
                self.steerer._last_probs = torch.cat(_probs_chunks, dim=1)
        else:
            # SDPA fast-path (수치적으로 원본과 동일)
            _sdpa_probs: List[torch.Tensor] = []
            for i in range(0, L_q, chunk_size):
                q_chunk = query[:, i:i + chunk_size, :]
                out_c   = F.scaled_dot_product_attention(
                    q_chunk, key, value, dropout_p=0.0, is_causal=False,
                )
                output[:, i:i + chunk_size, :] = out_c
                if _viz:
                    with torch.no_grad():
                        s_c = torch.bmm(
                            q_chunk.float(), key.float().transpose(-1, -2)
                        ) * attn.scale
                        _sdpa_probs.append(torch.softmax(s_c, dim=-1).cpu())
            if _viz and _sdpa_probs:
                self.steerer._last_probs = torch.cat(_sdpa_probs, dim=1)

        if _viz:
            self.steerer._last_gate = gate.detach().cpu()

        # ── Project back ────────────────────────────────────────────────────
        hidden_states_out = attn.batch_to_head_dim(output)
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
            scale=1.0,
            trust_scale=1.8,
            verify_scale=2.0,
            fusion_blocks="full",
        )

        with steerer.apply_steering():
            predictions = infer_batch(...)

    개별 효과 비활성화:
        scale=1.0, trust_scale=1.0, verify_scale=1.0  → 완전한 no-op
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
        self.force_verify  = bool(force_verify)
        self.fusion_blocks = fusion_blocks
        self.verbose       = verbose

        self.capture_for_viz: bool = False
        self._last_probs: Optional[torch.Tensor] = None
        self._last_gate:  Optional[torch.Tensor] = None

        self._active                   = False
        self._original_processors:     list = []
        self._steered_processors_pool: list = []
        self._attn_ref_modules:        list = []

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
        self._steered_processors_pool = [
            SteeredReferenceAttnProcessor(self)
            for _ in self._attn_ref_modules
        ]
        if self.verbose:
            print(f"[AICGSteerer] attn_ref modules collected: {len(self._attn_ref_modules)}")

    # ── processor swap context manager ──────────────────────────────────────

    @contextmanager
    def apply_steering(self):
        """attn_ref 모듈의 processor를 SteeredReferenceAttnProcessor로 교체.

        진입 시: 원본 processor 저장 → 교체.
        종료 시: 원본 복원 (예외 발생 시에도 보장).
        """
        if not self._attn_ref_modules:
            self._active = True
            try:
                yield self
            finally:
                self._active = False
            return

        self._original_processors = []
        for ar, steered in zip(self._attn_ref_modules, self._steered_processors_pool):
            self._original_processors.append(ar.processor)
            ar.set_processor(steered)

        if self.verbose:
            print(f"[AICGSteerer] steering ON  "
                  f"(trust_eff={self.effective_trust_scale:.3f}, "
                  f"verify_eff={self.effective_verify_scale:.3f})")

        self._active = True
        try:
            yield self
        finally:
            for ar, orig in zip(self._attn_ref_modules, self._original_processors):
                ar.set_processor(orig)
            self._active = False
            self._original_processors = []
            if self.verbose:
                print("[AICGSteerer] steering OFF (processors restored)")
