import torch
from torch import nn
from torch.nn import functional as F
from transformers import Lfm2Config, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .modules import Lfm2DecoderLayer, Lfm2RMSNorm, Lfm2RotaryEmbedding


class Lfm2PreTrainedModel(PreTrainedModel):
    config_class = Lfm2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["Lfm2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def __init__(self, config: Lfm2Config):
        super().__init__(config)

        if len(config.layer_types) != config.num_hidden_layers or any(
            layer_type not in {"conv", "full_attention"} for layer_type in config.layer_types
        ):
            raise ValueError("Set one valid layer type for each hidden layer.")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError(
                "The query head count must be divisible by the key/value head count."
            )


class Lfm2Model(Lfm2PreTrainedModel):
    def __init__(self, config: Lfm2Config):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            Lfm2DecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        )

        self.rotary_emb = Lfm2RotaryEmbedding(config)
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, config.norm_eps)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = False,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply either input_ids or inputs_embeds.")

        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        use_cache = self.config.use_cache if use_cache is None else use_cache

        if self.training and self.gradient_checkpointing:
            use_cache = False
            past_key_values = None

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            offset = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                torch.arange(hidden_states.shape[1], device=hidden_states.device)
                .unsqueeze(0)
                + offset
            )
        mask_arguments = dict(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        masks = {
            "full_attention": create_causal_mask(**mask_arguments),
            "conv": create_recurrent_attention_mask(**mask_arguments),
        }

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_type, layer in zip(self.config.layer_types, self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=masks[layer_type],
                past_key_values=past_key_values,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=self.embedding_norm(hidden_states),
            past_key_values=past_key_values,
        )


class Lfm2ForCausalLM(Lfm2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Lfm2Config):
        super().__init__(config)
        self.model = Lfm2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if logits_to_keep < 0 or (labels is not None and logits_to_keep):
            raise ValueError(
                "Use logits_to_keep=0 when labels are supplied, or a nonnegative value for inference."
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False if labels is not None and use_cache is None else use_cache,
        )

        logits = self.lm_head(outputs.last_hidden_state[:, -logits_to_keep:, :])
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                labels[:, 1:].to(logits.device).reshape(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
