import torch
from torch import nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from typing import Union, List

class Blip2VisionTower(nn.Module):
    """
    BLIP-2 Vision Tower, Text Embedding, Caption Generation 모듈

    예시:
        tower = Blip2VisionTower(
            model_name="Salesforce/blip2-flan-t5-xl",
            freeze=True
        )
        images = [Image.open("img1.jpg"), Image.open("img2.jpg")]
        texts = ["A photo of a cat.", "A photo of a dog."]
        img_feats, captions = tower(
            images=images,
            texts=texts,
            max_new_tokens=32,
            num_beams=5
        )
    """
    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        device: str = None
    ):
        super().__init__()
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if freeze:
            self.model.vision_model.requires_grad_(False)
            self.model.qformer.requires_grad_(False)

    def forward(
        self,
        images: Union[Image.Image, torch.Tensor, List[Image.Image]],
        texts: Union[str, List[str]] = None,
        max_new_tokens: int = 32,
        num_beams: int = 5,
        skip_special_tokens: bool = True
    ):
        """
        Args:
            images: PIL.Image 또는 torch.Tensor 혹은 리스트
            texts: 텍스트 프롬프트 (str 또는 List[str])
            max_new_tokens: 생성할 토큰 수
            num_beams: 빔 서치 빔 크기
        Returns:
            img_feats: torch.Tensor, [B, seq_len_img, hidden_size]
            captions: List[str]
        """
        # 리스트 일관화
        if not isinstance(images, (list, tuple)):
            images = [images]
        batch_size = len(images)
        # 텍스트 프롬프트 준비
        if texts is None:
            prompts = [""] * batch_size
        elif isinstance(texts, str):
            prompts = [texts]
        else:
            prompts = texts
        # do_rescale 결정
        first = images[0]
        do_rescale = not isinstance(first, torch.Tensor)
        # processor 호출 (이미지 + 텍스트)
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
        # 캡션 생성 - max_new_tokens 사용
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams
        }
        gen_ids = self.model.generate(**gen_kwargs)
        captions = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=skip_special_tokens
        )
        return img_feats, captions

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size
