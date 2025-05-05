# -*- coding: utf-8 -*-
"""
Multimodal Llava wrapper for OPT (causal LM) and T5 (seq2seq) using LlavaMetaModel & LlavaMetaForCausalLM
Based on the LlavaQwen implementation pattern
"""

import torch
import torch.nn as nn
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    OPTConfig, OPTForCausalLM, OPTModel,
    T5Config, T5ForConditionalGeneration, T5Model
)

from llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


# -----------------------------
# OPT 기반 Llava 모델
# -----------------------------
class LlavaOptConfig(OPTConfig):
    model_type = "llava_opt"


class LlavaOptModel(LlavaMetaModel, OPTModel):
    config_class = LlavaOptConfig

    def __init__(self, config: OPTConfig):
        super(LlavaOptModel, self).__init__(config)


class LlavaOptForCausalLM(OPTForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaOptConfig

    def __init__(self, config: OPTConfig):
        # Initialize OPTForCausalLM
        OPTForCausalLM.__init__(self, config)
        # Override model type
        config.model_type = "llava_opt"

        # LlavaMetaModel을 포함한 멀티모달 모델 선언
        self.model = LlavaOptModel(config)
        # 새로운 lm_head를 재선언하여 custom logits 계산
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
        # standard OPTForCausalLM forward
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        images=None,
        image_sizes=None,
        modalities=["image"],
        **kwargs
    ):
        # multimodal generate
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if images is not None:
            (inputs, position_ids, attention_mask,
             _, inputs_embeds, _) = \
                self.prepare_inputs_labels_for_multimodal(
                    inputs, position_ids, attention_mask,
                    None, None, images, modalities,
                    image_sizes=image_sizes
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

# Register to Auto classes
AutoConfig.register("llava_opt", LlavaOptConfig)
AutoModelForCausalLM.register(LlavaOptConfig, LlavaOptForCausalLM)


