import torch
import torch.nn as nn
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    T5Config, T5ForConditionalGeneration, T5Model
)

from llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
# -----------------------------
# T5 기반 Llava 모델 (Seq2Seq)
# -----------------------------
class LlavaT5Config(T5Config):
    model_type = "llava_t5"


class LlavaT5Model(LlavaMetaModel, T5Model):
    config_class = LlavaT5Config

    def __init__(self, config: T5Config):
        super(LlavaT5Model, self).__init__(config)


class LlavaT5ForConditionalGeneration(T5ForConditionalGeneration, LlavaMetaForCausalLM):
    config_class = LlavaT5Config

    def __init__(self, config: T5Config):
        # Initialize T5ForConditionalGeneration
        T5ForConditionalGeneration.__init__(self, config)
        config.model_type = "llava_t5"

        # LlavaMetaModel 포함
        self.model = LlavaT5Model(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
        modalities=["image"],
        **kwargs
    ):
        # multimodal 입력 전처리
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask,
             past_key_values, inputs_embeds, labels) = \
                self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask,
                    past_key_values, labels, images,
                    modalities, image_sizes
                )
        # T5 seq2seq forward (encoder-decoder)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

# Register T5 variant
auto_config = AutoConfig.register
auto_seq2seq = AutoModelForSeq2SeqLM.register

auto_config("llava_t5", LlavaT5Config)
auto_seq2seq(LlavaT5Config, LlavaT5ForConditionalGeneration)
