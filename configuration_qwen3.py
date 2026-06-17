"""
Configuration class for Qwen3-inspired MIL models.
Contains all hyperparameters for the gated attention and RMSNorm components.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Qwen3MILConfig:
    """
    Configuration for Qwen3-inspired MIL models.
    
    This config supports multiple ablation settings for paper experiments:
    - RMSNorm vs LayerNorm
    - Headwise vs Elementwise vs No gating
    - QK Normalization
    - Different attention patterns
    """
    
    # ============ Model Architecture ============
    in_dim: int = 1024
    """Input feature dimension from patch encoder"""
    
    embed_dim: int = 512
    """Internal embedding dimension"""
    
    num_classes: int = 1
    """Number of output classes (1 for binary)"""
    
    hidden_size: int = 512
    """Hidden size for transformer layers (usually same as embed_dim)"""
    
    intermediate_size: int = 2048
    """FFN intermediate size (typically 4x hidden_size)"""
    
    # ============ Attention Configuration ============
    num_attention_heads: int = 8
    """Number of attention heads"""
    
    num_key_value_heads: Optional[int] = None
    """Number of KV heads (None = same as attention heads, <num_heads for GQA)"""
    
    head_dim: Optional[int] = None
    """Dimension per head (None = hidden_size // num_attention_heads)"""
    
    attention_dropout: float = 0.0
    """Dropout rate for attention weights"""
    
    # ============ Gated Attention (Key Qwen3 Feature) ============
    headwise_attn_output_gate: bool = False
    """Use headwise gating (one gate per head) - Qwen3 style"""
    
    elementwise_attn_output_gate: bool = False
    """Use elementwise gating (one gate per element) - more expressive"""
    
    use_qk_norm: bool = True
    """Apply RMSNorm to Q and K before attention computation"""
    
    qkv_bias: bool = False
    """Whether to use bias in Q, K, V projections"""
    
    # ============ Normalization ============
    use_rms_norm: bool = True
    """Use RMSNorm instead of LayerNorm (Qwen3 style)"""
    
    rms_norm_eps: float = 1e-6
    """Epsilon for RMSNorm"""
    
    # ============ MLP Configuration ============
    hidden_act: str = "silu"
    """Activation function (silu/swish, gelu, relu)"""
    
    mlp_ratio: float = 4.0
    """MLP hidden dim ratio (intermediate_size = hidden_size * mlp_ratio)"""
    
    # ============ Regularization ============
    dropout: float = 0.25
    """General dropout rate"""
    
    # ============ MIL-Specific ============
    num_layers: int = 1
    """Number of transformer layers"""
    
    pooling: str = "attention"
    """Pooling type: 'attention', 'mean', 'max', 'cls'"""
    
    use_cls_token: bool = False
    """Whether to use a learnable CLS token"""
    
    # ============ Positional Encoding ============
    use_positional_encoding: bool = False
    """Whether to use positional encoding (usually False for MIL)"""
    
    max_position_embeddings: int = 8192
    """Maximum sequence length (for positional encoding if used)"""
    
    rope_theta: float = 10000.0
    """Base for RoPE encoding"""
    
    use_sliding_window: bool = False
    """Use sliding window attention"""
    
    sliding_window: Optional[int] = None
    """Sliding window size"""
    
    max_window_layers: int = 0
    """Number of layers to apply sliding window"""
    
    # ============ Implementation Details ============
    _attn_implementation: str = "sdpa"
    """Attention implementation: 'eager', 'sdpa', 'flash_attention_2'"""
    
    rope_scaling: Optional[dict] = None
    """RoPE scaling configuration"""
    
    def __post_init__(self):
        """Validate and set derived parameters."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        
        if self.intermediate_size is None or self.intermediate_size == 0:
            self.intermediate_size = int(self.hidden_size * self.mlp_ratio)
        
        # Ensure hidden_size matches embed_dim for consistency
        self.hidden_size = self.embed_dim
        
        # Validate gating options are mutually exclusive
        if self.headwise_attn_output_gate and self.elementwise_attn_output_gate:
            raise ValueError("Cannot use both headwise and elementwise attention gates")


# ============ Preset Configurations for Ablation Studies ============
GLOB_NLAYERS = 1
def get_qgmil_base_config(**kwargs) -> Qwen3MILConfig:
    """
    Base QGMIL configuration - all features enabled.
    Best performance expected.
    """
    defaults = dict(
        embed_dim=512,
        num_attention_heads=8,
        num_layers=GLOB_NLAYERS,
        use_rms_norm=True,
        use_qk_norm=True,
        headwise_attn_output_gate=True,
        elementwise_attn_output_gate=False,
        dropout=0.25,
        pooling="attention",
    )
    defaults.update(kwargs)
    return Qwen3MILConfig(**defaults)


def get_qgmil_elementwise_config(**kwargs) -> Qwen3MILConfig:
    """
    QGMIL with elementwise gating instead of headwise.
    More parameters, potentially more expressive.
    """
    defaults = dict(
        embed_dim=512,
        num_attention_heads=8,
        num_layers=GLOB_NLAYERS,
        use_rms_norm=True,
        use_qk_norm=True,
        headwise_attn_output_gate=False,
        elementwise_attn_output_gate=True,
        dropout=0.25,
        pooling="attention",
    )
    defaults.update(kwargs)
    return Qwen3MILConfig(**defaults)


def get_qgmil_no_gate_config(**kwargs) -> Qwen3MILConfig:
    """
    QGMIL without any gating mechanism.
    Ablation: shows impact of gating.
    """
    defaults = dict(
        embed_dim=512,
        num_attention_heads=8,
        num_layers=GLOB_NLAYERS,
        use_rms_norm=True,
        use_qk_norm=True,
        headwise_attn_output_gate=False,
        elementwise_attn_output_gate=False,
        dropout=0.25,
        pooling="attention",
    )
    defaults.update(kwargs)
    return Qwen3MILConfig(**defaults)


def get_qgmil_layernorm_config(**kwargs) -> Qwen3MILConfig:
    """
    QGMIL with LayerNorm instead of RMSNorm.
    Ablation: shows impact of RMSNorm.
    """
    defaults = dict(
        embed_dim=512,
        num_attention_heads=8,
        num_layers=GLOB_NLAYERS,
        use_rms_norm=False,
        use_qk_norm=False,
        headwise_attn_output_gate=True,
        elementwise_attn_output_gate=False,
        dropout=0.25,
        pooling="attention",
    )
    defaults.update(kwargs)
    return Qwen3MILConfig(**defaults)


def get_qgmil_no_qknorm_config(**kwargs) -> Qwen3MILConfig:
    """
    QGMIL without QK normalization.
    Ablation: shows impact of QK norm.
    """
    defaults = dict(
        embed_dim=512,
        num_attention_heads=8,
        num_layers=GLOB_NLAYERS,
        use_rms_norm=True,
        use_qk_norm=False,
        headwise_attn_output_gate=True,
        elementwise_attn_output_gate=False,
        dropout=0.25,
        pooling="attention",
    )
    defaults.update(kwargs)
    return Qwen3MILConfig(**defaults)


def get_qgmil_minimal_config(**kwargs) -> Qwen3MILConfig:
    """
    Minimal QGMIL - no special features.
    Ablation baseline: standard transformer MIL.
    """
    defaults = dict(
        embed_dim=512,
        num_attention_heads=8,
        num_layers=GLOB_NLAYERS,
        use_rms_norm=False,
        use_qk_norm=False,
        headwise_attn_output_gate=False,
        elementwise_attn_output_gate=False,
        dropout=0.25,
        pooling="attention",
    )
    defaults.update(kwargs)
    return Qwen3MILConfig(**defaults)


def get_qgmil_deep_config(**kwargs) -> Qwen3MILConfig:
    """
    Deeper QGMIL with more layers.
    For capacity ablation.
    """
    defaults = dict(
        embed_dim=512,
        num_attention_heads=8,
        num_layers=4,
        use_rms_norm=True,
        use_qk_norm=True,
        headwise_attn_output_gate=True,
        elementwise_attn_output_gate=False,
        dropout=0.25,
        pooling="attention",
    )
    defaults.update(kwargs)
    return Qwen3MILConfig(**defaults)


def get_qgmil_light_config(**kwargs) -> Qwen3MILConfig:
    """
    Lightweight QGMIL for efficiency comparison.
    """
    defaults = dict(
        embed_dim=256,
        num_attention_heads=4,
        num_layers=1,
        use_rms_norm=True,
        use_qk_norm=True,
        headwise_attn_output_gate=True,
        elementwise_attn_output_gate=False,
        dropout=0.25,
        pooling="attention",
    )
    defaults.update(kwargs)
    return Qwen3MILConfig(**defaults)


# Registry of all configurations for easy access
QGMIL_CONFIGS = {
    "base": get_qgmil_base_config,
    "elementwise": get_qgmil_elementwise_config,
    "no_gate": get_qgmil_no_gate_config,
    "layernorm": get_qgmil_layernorm_config,
    "no_qknorm": get_qgmil_no_qknorm_config,
    "minimal": get_qgmil_minimal_config,
    "deep": get_qgmil_deep_config,
    "light": get_qgmil_light_config,
}


def get_config(name: str = "base", **kwargs) -> Qwen3MILConfig:
    """
    Get a configuration by name.
    
    Args:
        name: Configuration name from QGMIL_CONFIGS
        **kwargs: Override any config parameter
    
    Returns:
        Qwen3MILConfig instance
    """
    if name not in QGMIL_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(QGMIL_CONFIGS.keys())}")
    return QGMIL_CONFIGS[name](**kwargs)
