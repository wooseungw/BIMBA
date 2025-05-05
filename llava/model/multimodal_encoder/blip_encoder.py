import torch
from torch import nn
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from typing import Union, List
from llava.utils import rank0_print




class Blip2VisionTower(nn.Module):
    """
    BLIP-2 Vision Tower, Text Embedding, Caption Generation 모듈
    사용 예시:
        tower = Blip2VisionTower(
            model_name="Salesforce/blip2-flan-t5-xl",
            freeze=True
        )
        images = [Image.open("img1.jpg"), Image.open("img2.jpg")]
        texts = ["A photo of a cat.", "A photo of a dog."]
        img_feats, txt_embs, captions = tower(
            images=images,
            texts=texts
        )
        # img_feats: torch.Tensor [B, seq_len_img, hidden_size]
        # captions: List[str]
    """
    def __init__(
        self,
        model_name: str,
        vision_tower_cfg,
        delay_load=False
    ):
        super().__init__()
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        
        if not delay_load:
            rank0_print(f"Loading vision tower: {model_name}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def forward(
        self,
        images: Union[Image.Image, torch.Tensor, List[Image.Image]],
        query: Union[str, List[str]] = None,
        max_length: int = 32,
        num_beams: int = 5,
        skip_special_tokens: bool = True
    ):
        # 리스트 일관화
        if not isinstance(images, (list, tuple)):
            images = [images]
        batch_size = len(images)
        # 텍스트 프롬프트 준비
        if query is None:
            prompts = [""] * batch_size
        elif isinstance(query, str):
            prompts = [query]
        else:
            prompts = query
        # do_rescale 결정
        first = images[0]
        do_rescale = not isinstance(first, torch.Tensor)
        # processor 호출 (이미지+텍스트)
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            do_rescale=do_rescale
        ).to(self.device)
        # 이미지 특징 추출
        vision_outputs = self.model.vision_model(
            pixel_values=inputs.pixel_values,
            output_hidden_states=True,
            return_dict=True
        )
        img_feats = vision_outputs.hidden_states[-1]
        
        # 캡션 생성
        gen_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams
        )
        captions = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=skip_special_tokens
        )
        return {"image_features": img_feats, "captions": captions}

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size
