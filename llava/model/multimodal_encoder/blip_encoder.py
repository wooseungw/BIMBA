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
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config
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
        vision_tower: str,
        vision_tower_cfg: Blip2Config = Blip2Config(),
        delay_load=False
    ):
        super().__init__()
        
        self.vision_tower_name = vision_tower
        self.config = Blip2Config()
        self.processor = Blip2Processor.from_pretrained(self.vision_tower_name)
        if vision_tower:
            self.vision_tower = Blip2ForConditionalGeneration.from_pretrained(self.vision_tower_name, device_map="auto")
        else:
            self.cfg_only = self.config
            self.vision_tower = Blip2ForConditionalGeneration(vision_tower_cfg)
        self.vision_tower.eval()
    
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
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            do_rescale=do_rescale
        ).to(self.vision_tower.device)
        image_features =[]
        captions = []
        if type(images) is list:
            for image in images:
                # 이미지 특징 추출
                vision_outputs = self.vision_tower.vision_model(
                    pixel_values=inputs.pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                img_feats = vision_outputs.hidden_states[-1]
                image_features.append(img_feats)
                # 캡션 생성
                gen_ids = self.vision_tower.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=num_beams
                )
                captions = self.processor.batch_decode(
                    gen_ids,
                    skip_special_tokens=skip_special_tokens
                )
                image_features.append(img_feats)
                captions.append(captions)
                
        return  image_features, captions
    

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size
