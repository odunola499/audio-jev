import torch
from torch import nn
from transformers import PreTrainedConfig

from audio_jev.text.model.lfm import Lfm2Model, Lfm2Config
from audio_jev.text.model.jev_head import JointSchemaHead, EncodedRecord


class JevModelConfig(PreTrainedConfig):
    model_type = "lfm2_jev"

    def __init__(
        self,
        base_model_config=None,
        width=256,
        routing_layers=2,
        layers=4,
        heads=4,
        feedforward=1024,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(base_model_config, dict):
            base_model_config = Lfm2Config(**base_model_config)

        self.base_model_config = base_model_config or Lfm2Config()
        self.hidden_size = self.base_model_config.hidden_size
        self.width = width
        self.routing_layers = routing_layers
        self.layers = layers
        self.heads = heads
        self.feedforward = feedforward
        self.dropout = dropout


class JevModel(nn.Module):
    def __init__(self, config: JevModelConfig, base: Lfm2Model | None = None):
        super().__init__()
        self.config = config
        self.base = base if base is not None else Lfm2Model(config.base_model_config)
        self.head = JointSchemaHead(
            hidden_size=self.config.hidden_size,
            width=self.config.width,
            routing_layers=config.routing_layers,
            layers=config.layers,
            heads=config.heads,
            feedforward=config.feedforward,
            dropout=config.dropout,
        )

    @property
    def output_embedding_weight(self):
        return self.base.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        records: list[EncodedRecord],
        attention_mask: torch.Tensor | None = None,
    ):
        hidden_states = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state

        return self.head(
            hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            records=records,
            output_embedding_weight=self.output_embedding_weight,
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from audio_jev.text.examples import EXAMPLE_REQUEST
    from audio_jev.text.model.utilities import encode_prompt, collate_prompts

    lfm_config = Lfm2Config()
    config = JevModelConfig(
        base_model_config = lfm_config,
    )
    model = JevModel(config)
    tokenizer = AutoTokenizer.from_pretrained(
        "LiquidAI/LFM2.5-350M"
    )

    example = EXAMPLE_REQUEST
    record = encode_prompt(
        tokenizer = tokenizer,
        prompt = example)
    print(record)

    input_ids, attention_mask = collate_prompts(
        records = [record],
        pad_token_id=7
    )

    model_output = model(
        input_ids = input_ids,
        attention_mask = attention_mask,
        records = [record],
    )
    print(model_output)

