import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config


class Lfm2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = hidden_states.float()
        values = values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + self.variance_epsilon
        )

        return self.weight * values.to(hidden_states.dtype)


class Lfm2RotaryEmbedding(nn.Module):
    def __init__(self, config: Lfm2Config):
        super().__init__()

        if config.rope_parameters["rope_type"] != "default":
            raise ValueError(
                "Only the original, unscaled LFM2 rotary embeddings are supported."
            )

        self.head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        self.theta = config.rope_parameters["rope_theta"]

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            frequencies = 1.0 / self.theta ** (
                torch.arange(
                    0, self.head_dim, 2,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                / self.head_dim
            )

            angles = position_ids.float().unsqueeze(-1) * frequencies
            angles = torch.cat((angles, angles), dim=-1)

            return angles.cos().to(hidden_states.dtype), angles.sin().to(hidden_states.dtype)


def apply_rotary_pos_emb(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)

    return values * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


class Lfm2MLP(nn.Module):
    def __init__(self, config: Lfm2Config):
        super().__init__()

        intermediate_size = config.intermediate_size

        if config.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)

            if config.block_ffn_dim_multiplier is not None:
                intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
                multiple = config.block_multiple_of
                intermediate_size = multiple * ((intermediate_size + multiple - 1) // multiple)

        self.w1 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class Lfm2Attention(nn.Module):
    def __init__(self, config: Lfm2Config, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx

        self.head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        query_size = config.num_attention_heads * self.head_dim
        key_value_size = config.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, query_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.out_proj = nn.Linear(query_size, config.hidden_size, bias=False)

        self.q_layernorm = Lfm2RMSNorm(self.head_dim, config.norm_eps)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim, config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        shape = (batch_size, sequence_length, -1, self.head_dim)

        query = self.q_layernorm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        key = self.k_layernorm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)

        cos, sin = position_embeddings
        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=attention_mask is None and sequence_length > 1,
        )

        return self.out_proj(output.transpose(1, 2).reshape(batch_size, sequence_length, -1))


class Lfm2ShortConv(nn.Module):
    def __init__(self, config: Lfm2Config, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        self.conv_kernel_size = config.conv_L_cache

        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=self.conv_kernel_size,
            groups=config.hidden_size,
            bias=config.conv_bias,
            padding=self.conv_kernel_size - 1,
        )

        self.in_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.conv_bias
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.conv_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[1]

        if attention_mask is not None:
            hidden_states = (hidden_states * attention_mask.unsqueeze(-1)).to(
                hidden_states.dtype
            )

        input_gate, output_gate, values = (
            self.in_proj(hidden_states).transpose(1, 2).chunk(3, dim=1)
        )
        values = input_gate * values

        if past_key_values is not None:
            values = past_key_values.update_conv_state(
                values, self.layer_idx, conv_kernel_size=self.conv_kernel_size
            )

        output = self.conv(values.to(self.conv.weight.dtype))[..., : values.shape[-1]]
        output = output[..., -sequence_length:].to(hidden_states.dtype)

        return self.out_proj((output_gate * output).transpose(1, 2).contiguous())


class Lfm2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm2Config, layer_idx: int):
        super().__init__()

        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"

        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ShortConv(config, layer_idx)

        self.feed_forward = Lfm2MLP(config)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        normalized = self.operator_norm(hidden_states)

        if self.is_attention_layer:
            output = self.self_attn(
                normalized, position_embeddings, attention_mask, past_key_values
            )
        else:
            output = self.conv(normalized, past_key_values, attention_mask)

        hidden_states = hidden_states + output

        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
