"""
QGMIL: Qwen3-inspired Gated Multiple Instance Learning Models.

This module implements MIL models with key architectural innovations from Qwen3:
1. RMSNorm for better training stability
2. Gated attention output (headwise or elementwise)
3. QK normalization for improved attention
4. SwiGLU-style MLP

These models are designed for Whole Slide Image (WSI) classification tasks
and are compatible with the existing training pipeline.
"""
# This file contains components adapted in part from the Qwen open-source implementation:
# https://github.com/QwenLM/Qwen
# Modifications were made for Multiple Instance Learning in medical imaging.
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_qwen3 import Qwen3MILConfig, get_config


# ============================================================================
# Core Components from Qwen3
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    Simpler and often more effective than LayerNorm.
    From: https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLUMLP(nn.Module):
    """
    SwiGLU-style MLP as used in Qwen3/LLaMA.
    
    Uses gated activation: out = down(act(gate(x)) * up(x))
    """
    
    def __init__(self, config: Qwen3MILConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Activation function
        if config.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif config.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif config.hidden_act == "relu":
            self.act_fn = nn.ReLU()
        else:
            self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class StandardMLP(nn.Module):
    """Standard MLP without gating for ablation."""
    
    def __init__(self, config: Qwen3MILConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
        if config.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif config.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        else:
            self.act_fn = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


# ============================================================================
# Gated Attention Module (Key Innovation)
# ============================================================================

class GatedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional gating mechanisms from Qwen3.
    
    Supports:
    - Headwise gating: one learnable gate per attention head
    - Elementwise gating: one learnable gate per element
    - QK normalization: RMSNorm on queries and keys
    """
    
    def __init__(self, config: Qwen3MILConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Gating options
        self.headwise_attn_output_gate = config.headwise_attn_output_gate
        self.elementwise_attn_output_gate = config.elementwise_attn_output_gate
        self.use_qk_norm = config.use_qk_norm
        
        # Scaling factor
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)
        
        # Projections with optional extra dimensions for gating
        if self.headwise_attn_output_gate:
            # Extra dimension for headwise gates
            self.q_proj = nn.Linear(
                self.hidden_size, 
                self.num_heads * self.head_dim + self.num_heads, 
                bias=config.qkv_bias
            )
        elif self.elementwise_attn_output_gate:
            # Double Q projection for elementwise gates
            self.q_proj = nn.Linear(
                self.hidden_size, 
                self.num_heads * self.head_dim * 2, 
                bias=config.qkv_bias
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, 
                self.num_heads * self.head_dim, 
                bias=config.qkv_bias
            )
        
        self.k_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.qkv_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.qkv_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, 
            self.hidden_size, 
            bias=config.qkv_bias
        )
        
        # QK normalization
        if self.use_qk_norm:
            if config.use_rms_norm:
                self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
                self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            else:
                self.q_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)
                self.k_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional mask (batch, 1, seq_len, seq_len)
            output_attentions: Whether to return attention weights
        
        Returns:
            output: (batch, seq_len, hidden_size)
            attention_weights: Optional (batch, num_heads, seq_len, seq_len)
        """
        bsz, seq_len, _ = hidden_states.size()
        
        # Compute Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Handle gating
        gate_score = None
        if self.headwise_attn_output_gate:
            query_states = query_states.view(bsz, seq_len, self.num_key_value_heads, -1)
            query_states, gate_score = torch.split(
                query_states, 
                [self.head_dim * self.num_key_value_groups, self.num_key_value_groups], 
                dim=-1
            )
            gate_score = gate_score.reshape(bsz, seq_len, -1, 1)
            query_states = query_states.reshape(bsz, seq_len, -1, self.head_dim).transpose(1, 2)
        elif self.elementwise_attn_output_gate:
            query_states = query_states.view(bsz, seq_len, self.num_key_value_heads, -1)
            query_states, gate_score = torch.split(
                query_states, 
                [self.head_dim * self.num_key_value_groups, self.head_dim * self.num_key_value_groups], 
                dim=-1
            )
            gate_score = gate_score.reshape(bsz, seq_len, -1, self.head_dim)
            query_states = query_states.reshape(bsz, seq_len, -1, self.head_dim).transpose(1, 2)
        else:
            query_states = query_states.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2)
        
        # Reshape K and V
        key_states = key_states.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2)
        
        # Apply QK normalization
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        
        # Repeat KV heads if using GQA
        if self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.inv_sqrt_head_dim
        
        # Apply mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_dropped = self.attention_dropout(attn_weights)
        
        # Compute output
        attn_output = torch.matmul(attn_weights_dropped, value_states)
        
        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Apply gating
        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)
        
        attn_output = attn_output.reshape(bsz, seq_len, -1)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None
    
    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for Grouped Query Attention."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ============================================================================
# Transformer Block
# ============================================================================

class QGMILBlock(nn.Module):
    """
    Single transformer block for QGMIL.
    
    Pre-norm architecture with:
    - RMSNorm/LayerNorm
    - Gated Multi-Head Attention
    - SwiGLU MLP
    - Residual connections
    """
    
    def __init__(self, config: Qwen3MILConfig):
        super().__init__()
        self.config = config
        
        # Normalization layers
        if config.use_rms_norm:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Attention
        self.self_attn = GatedMultiHeadAttention(config)
        
        # MLP
        self.mlp = SwiGLUMLP(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            output_attentions: Return attention weights
        
        Returns:
            hidden_states: (batch, seq_len, hidden_size)
            attention_weights: Optional
        """
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights


# ============================================================================
# Attention Pooling for MIL
# ============================================================================

class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Pooling for MIL aggregation.
    
    Computes attention weights over all instances and performs weighted sum.
    Uses gating mechanism similar to ABMIL but enhanced with Qwen3 components.
    """
    
    def __init__(self, config: Qwen3MILConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(config.hidden_size // 2, 1)
        
        # Optional: Normalization before pooling
        if config.use_rms_norm:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            mask: Optional mask for padding
        
        Returns:
            pooled: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        hidden_states = self.norm(hidden_states)
        
        # Compute gated attention
        A_V = self.attention_V(hidden_states)  # (batch, seq_len, hidden_size // 2)
        A_U = self.attention_U(hidden_states)  # (batch, seq_len, hidden_size // 2)
        A = self.attention_weights(A_V * A_U)  # (batch, seq_len, 1)
        A = A.squeeze(-1)  # (batch, seq_len)
        
        # Apply mask
        if mask is not None:
            A = A.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        A = F.softmax(A, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        pooled = torch.bmm(A.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_size)
        
        return pooled, A


# ============================================================================
# Main QGMIL Model
# ============================================================================

class QGMIL(nn.Module):
    """
    Qwen3-inspired Gated Multiple Instance Learning Model.
    
    Architecture:
    1. Input projection (patch features -> hidden_size)
    2. Optional CLS token
    3. N transformer blocks with gated attention
    4. Attention pooling
    5. Classification head
    
    Key features:
    - RMSNorm for stability
    - Gated attention output
    - QK normalization
    - SwiGLU MLP
    """
    
    def __init__(self, config: Qwen3MILConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.in_dim, config.hidden_size),
            RMSNorm(config.hidden_size) if config.use_rms_norm else nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Optional CLS token
        self.use_cls_token = config.use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            QGMILBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final normalization
        if config.use_rms_norm:
            self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.final_norm = nn.LayerNorm(config.hidden_size)
        
        # Pooling
        self.pooling_type = config.pooling
        if self.pooling_type == "attention":
            self.pooling = GatedAttentionPooling(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, num_patches, in_dim)
            mask: Optional padding mask (batch, num_patches)
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with 'logits' and optionally 'attention'
        """
        # Handle 2D input (single sample without batch)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size, num_patches, _ = x.shape
        
        # Input projection
        hidden_states = self.input_proj(x)
        
        # Add CLS token if configured
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)
            num_patches += 1
        
        # Create attention mask (for padding if needed)
        attention_mask = None
        if mask is not None:
            if self.use_cls_token:
                # Add 1 for CLS token
                cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)
            # Convert to attention mask format: (batch, 1, 1, seq_len)
            attention_mask = (1.0 - mask.unsqueeze(1).unsqueeze(2)) * torch.finfo(hidden_states.dtype).min
        
        # Pass through transformer blocks
        all_attentions = []
        for layer in self.layers:
            hidden_states, attn_weights = layer(
                hidden_states, 
                attention_mask=attention_mask,
                output_attentions=return_attention
            )
            if return_attention and attn_weights is not None:
                all_attentions.append(attn_weights)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Pooling
        if self.use_cls_token:
            # Use CLS token
            pooled = hidden_states[:, 0]
            pool_attention = None
        elif self.pooling_type == "attention":
            pooled, pool_attention = self.pooling(hidden_states, mask)
        elif self.pooling_type == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                pooled = (hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1)
            else:
                pooled = hidden_states.mean(dim=1)
            pool_attention = None
        elif self.pooling_type == "max":
            pooled = hidden_states.max(dim=1)[0]
            pool_attention = None
        else:
            pooled = hidden_states.mean(dim=1)
            pool_attention = None
        
        # Classification
        logits = self.classifier(pooled)
        
        # Prepare output
        output = {"logits": logits}
        
        if return_attention:
            output["layer_attentions"] = all_attentions
            output["pool_attention"] = pool_attention
        
        return output


# ============================================================================
# Model Variants for Different Ablations
# ============================================================================

class QGMILBase(QGMIL):
    """QGMIL with all features enabled (headwise gating, RMSNorm, QK norm)."""
    
    def __init__(self, in_dim: int, embed_dim: int = 512, dropout: float = 0.25, num_classes: int = 1, **kwargs):
        config = get_config("base", in_dim=in_dim, embed_dim=embed_dim, dropout=dropout, num_classes=num_classes, **kwargs)
        super().__init__(config)


class QGMILElementwise(QGMIL):
    """QGMIL with elementwise gating."""
    
    def __init__(self, in_dim: int, embed_dim: int = 512, dropout: float = 0.25, num_classes: int = 1, **kwargs):
        config = get_config("elementwise", in_dim=in_dim, embed_dim=embed_dim, dropout=dropout, num_classes=num_classes, **kwargs)
        super().__init__(config)


class QGMILNoGate(QGMIL):
    """QGMIL without gating (ablation)."""
    
    def __init__(self, in_dim: int, embed_dim: int = 512, dropout: float = 0.25, num_classes: int = 1, **kwargs):
        config = get_config("no_gate", in_dim=in_dim, embed_dim=embed_dim, dropout=dropout, num_classes=num_classes, **kwargs)
        super().__init__(config)


class QGMILLayerNorm(QGMIL):
    """QGMIL with LayerNorm instead of RMSNorm (ablation)."""
    
    def __init__(self, in_dim: int, embed_dim: int = 512, dropout: float = 0.25, num_classes: int = 1, **kwargs):
        config = get_config("layernorm", in_dim=in_dim, embed_dim=embed_dim, dropout=dropout, num_classes=num_classes, **kwargs)
        super().__init__(config)


class QGMILNoQKNorm(QGMIL):
    """QGMIL without QK normalization (ablation)."""
    
    def __init__(self, in_dim: int, embed_dim: int = 512, dropout: float = 0.25, num_classes: int = 1, **kwargs):
        config = get_config("no_qknorm", in_dim=in_dim, embed_dim=embed_dim, dropout=dropout, num_classes=num_classes, **kwargs)
        super().__init__(config)


class QGMILMinimal(QGMIL):
    """Minimal QGMIL - standard transformer without Qwen3 features (ablation baseline)."""
    
    def __init__(self, in_dim: int, embed_dim: int = 512, dropout: float = 0.25, num_classes: int = 1, **kwargs):
        config = get_config("minimal", in_dim=in_dim, embed_dim=embed_dim, dropout=dropout, num_classes=num_classes, **kwargs)
        super().__init__(config)


class QGMILDeep(QGMIL):
    """Deep QGMIL with 4 layers."""
    
    def __init__(self, in_dim: int, embed_dim: int = 512, dropout: float = 0.25, num_classes: int = 1, **kwargs):
        config = get_config("deep", in_dim=in_dim, embed_dim=embed_dim, dropout=dropout, num_classes=num_classes, **kwargs)
        super().__init__(config)


class QGMILLight(QGMIL):
    """Lightweight QGMIL for efficiency."""
    
    def __init__(self, in_dim: int, embed_dim: int = 256, dropout: float = 0.25, num_classes: int = 1, **kwargs):
        config = get_config("light", in_dim=in_dim, embed_dim=embed_dim, dropout=dropout, num_classes=num_classes, **kwargs)
        super().__init__(config)


# ============================================================================
# Factory and Registry
# ============================================================================

QGMIL_MODELS = {
    "qgmil": QGMILBase,
    "qgmil_base": QGMILBase,
    "qgmil_elementwise": QGMILElementwise,
    "qgmil_no_gate": QGMILNoGate,
    "qgmil_layernorm": QGMILLayerNorm,
    "qgmil_no_qknorm": QGMILNoQKNorm,
    "qgmil_minimal": QGMILMinimal,
    "qgmil_deep": QGMILDeep,
    "qgmil_light": QGMILLight,
}


def get_qgmil_model(
    model_type: str = "qgmil",
    in_dim: int = 1024,
    embed_dim: int = 512,
    dropout: float = 0.25,
    num_classes: int = 1,
    **kwargs
) -> QGMIL:
    """
    Get a QGMIL model by type.
    
    Args:
        model_type: Model variant name
        in_dim: Input feature dimension
        embed_dim: Embedding dimension
        dropout: Dropout rate
        num_classes: Number of output classes
        **kwargs: Additional config overrides
    
    Returns:
        QGMIL model instance
    """
    if model_type not in QGMIL_MODELS:
        raise ValueError(f"Unknown QGMIL model: {model_type}. Available: {list(QGMIL_MODELS.keys())}")
    
    return QGMIL_MODELS[model_type](
        in_dim=in_dim,
        embed_dim=embed_dim,
        dropout=dropout,
        num_classes=num_classes,
        **kwargs
    )
