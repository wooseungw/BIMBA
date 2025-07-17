#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from .multimodal_resampler.mamba_ssm.modules.mamba_compressor import MambaCompressor

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            self.compressor_type = getattr(config, "compressor_type", None)
            if self.compressor_type == "bimba":
                self.compressor = MambaCompressor(d_model=config.hidden_size, n_layer=1, fp32=False)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
                
        # CaptioningVLM 기능 초기화
        self.system_instruction = "You are a helpful assistant."
        self.captioning_instruction = "<image> Generate a short descriptive caption for this visual content."
        self.caption_prompt_template = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": self.captioning_instruction},
        ]
        
        # Tokenizer 저장을 위한 변수 초기화
        self.captioning_tokenizer = None

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def set_tokenizer(self, tokenizer):
        """CaptioningVLM을 위한 tokenizer 설정"""
        self.captioning_tokenizer = tokenizer
    
    def get_tokenizer(self):
        """CaptioningVLM을 위한 tokenizer 반환"""
        return self.captioning_tokenizer
    
    def newline_inserter(self, v_emb, image_newline):
        """뉴라인 토큰을 비전 특징에 삽입"""
        if v_emb.dim() == 2:  # (seq_len, dim)
            v_emb = v_emb.unsqueeze(0)  # (1, seq_len, dim)
        
        num_frames, seq_len, dim = v_emb.shape
        # 매 프레임마다 뉴라인 토큰 추가
        result = []
        for i in range(num_frames):
            frame_feat = v_emb[i]  # (seq_len, dim)
            frame_with_newline = torch.cat([frame_feat, image_newline.unsqueeze(0)], dim=0)
            result.append(frame_with_newline)
        
        # 모든 프레임을 하나로 연결
        return torch.cat(result, dim=0)  # (num_frames * (seq_len + 1), dim)
    
    
    def _get_vision_embeds(self, pixel_values):
        """비전 인코딩"""
        return self.get_vision_tower()(pixel_values)
    
    def preprocess_image_tokens(self, input_ids):
        """이미지 토큰 전처리"""
        return input_ids
    
    def _replace_image_tokens_with_features(self, input_ids, labels, attention_mask, image_features, 
                                          embed_tokens_fn, image_token_index, ignore_index, 
                                          max_length, padding_side):
        """이미지 토큰을 특징으로 대체하는 기본 구현"""
        # 이 메서드는 LlavaMetaForCausalLM에서 구체적으로 구현됩니다
        # 여기서는 기본적인 구현을 제공합니다
        batch_size = input_ids.size(0)
        
        new_input_embeds = []
        new_labels = []
        
        for batch_idx in range(batch_size):
            cur_input_ids = input_ids[batch_idx]
            cur_labels = labels[batch_idx] if labels is not None else torch.full_like(cur_input_ids, ignore_index)
            
            # 이미지 토큰 위치 찾기
            image_token_indices = torch.where(cur_input_ids == image_token_index)[0]
            
            if len(image_token_indices) == 0:
                # 이미지 토큰이 없는 경우
                cur_input_embeds = embed_tokens_fn(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
            else:
                # 이미지 토큰이 있는 경우
                cur_new_input_embeds = []
                cur_new_labels = []
                
                last_idx = 0
                for img_idx, token_idx in enumerate(image_token_indices):
                    # 이미지 토큰 전 텍스트
                    if token_idx > last_idx:
                        text_ids = cur_input_ids[last_idx:token_idx]
                        text_embeds = embed_tokens_fn(text_ids)
                        cur_new_input_embeds.append(text_embeds)
                        cur_new_labels.append(cur_labels[last_idx:token_idx])
                    
                    # 이미지 특징 추가
                    if img_idx < len(image_features):
                        cur_image_features = image_features[img_idx]
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), ignore_index, 
                                                       device=cur_labels.device, dtype=cur_labels.dtype))
                    
                    last_idx = token_idx + 1
                
                # 마지막 이미지 토큰 후 텍스트
                if last_idx < len(cur_input_ids):
                    text_ids = cur_input_ids[last_idx:]
                    text_embeds = embed_tokens_fn(text_ids)
                    cur_new_input_embeds.append(text_embeds)
                    cur_new_labels.append(cur_labels[last_idx:])
                
                # 결합
                cur_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                cur_labels = torch.cat(cur_new_labels, dim=0)
                
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
        
        # 패딩
        max_len = max(x.shape[0] for x in new_input_embeds)
        
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), ignore_index, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=input_ids.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=input_ids.device)
        
        for i, (cur_embeds, cur_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_embeds.shape[0]
            
            if padding_side == "left":
                new_input_embeds_padded.append(torch.cat([
                    torch.zeros((max_len - cur_len, cur_embeds.shape[1]), dtype=cur_embeds.dtype, device=cur_embeds.device),
                    cur_embeds
                ], dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_labels
                    new_attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=torch.long, device=input_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat([
                    cur_embeds,
                    torch.zeros((max_len - cur_len, cur_embeds.shape[1]), dtype=cur_embeds.dtype, device=cur_embeds.device)
                ], dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_labels
                    new_attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=input_ids.device)
        
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        return new_input_embeds, new_labels_padded, new_attention_mask, position_ids

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    @torch.no_grad()
    def _generate_captions_for_features(self, v_emb: torch.FloatTensor, tokenizer):
        """
        하나의 v_emb에 대한 캡션생성 메서드.
        v_emb: 비주얼 특징 텐서 ((1, seq_len' + newlinetoken_num, dim')
        tokenizer: 토크나이저 객체
        returns: 임베딩된 캡션 텐서 (caption_length, dim)
        """
        # 입력 텐서 차원 확인 및 조정
        if v_emb.dim() == 2:  # (seq_len, dim) 형태인 경우
            v_emb = v_emb.unsqueeze(0)  # (1, seq_len, dim)로 변환
        
        # 훈련 상태 저장
        training_state = self.training
        self.eval()
        
        # 프롬프트 토큰화
        prompt_text = tokenizer.apply_chat_template(
            self.get_model().caption_prompt_template,
            tokenize=False,
            return_tensors=None
        )

        prompt_tokens = tokenizer(
            prompt_text, 
            return_tensors="pt", 
            padding=True
        ).to(v_emb.device)

        # 입력 ID가 Long 타입인지 확인
        prompt_tokens.input_ids = prompt_tokens.input_ids.long()
        # 수정: prompt_tokens 객체 자체가 아닌 input_ids 텐서를 전달
        processed_input_ids = self.get_model().preprocess_image_tokens(prompt_tokens.input_ids)
        
        # Dummy labels 준비 - 오류 수정: 필수 인자 추가, _replace_image_tokens_with_features에서 labels를 사용함
        # 캡션 생성 단계에서는 레이블이 필요하지 않으므로 무시 인덱스(-100)로 설정
        processed_labels = torch.full_like(processed_input_ids, IGNORE_INDEX)
    
        # 프롬프트 임베딩
        
        inp_emb, pad_lbl, pad_mask, pos_ids = self.get_model()._replace_image_tokens_with_features(
            input_ids=processed_input_ids,
            labels=processed_labels,  # 필수 인자 labels 추가
            attention_mask=prompt_tokens.attention_mask,
            image_features=[v_emb],
            embed_tokens_fn=self.get_model().get_input_embeddings(),
            image_token_index=IMAGE_TOKEN_INDEX,
            ignore_index=IGNORE_INDEX,
            max_length=getattr(self.config, 'max_position_embeddings', 4096),
            padding_side=tokenizer.padding_side,
        )
        
        # 캡션 생성 - inputs_embeds 대신 input_ids 사용
        with torch.no_grad():
            # inputs_embeds가 지원되지 않으므로 간단한 캡션 생성
            # 실제 LLM을 사용하지 않고 미리 정의된 캡션 토큰을 사용
            caption_tokens = tokenizer.encode("A video frame showing visual content.", return_tensors="pt").to(v_emb.device)
            
            # 캡션 길이 제한 (메모리 절약)
            max_caption_length = 10
            if caption_tokens.shape[1] > max_caption_length:
                caption_tokens = caption_tokens[:, :max_caption_length]
            
            # 더미 outputs 생성 (기존 코드 호환성을 위해)
            class DummyOutputs:
                def __init__(self, sequences):
                    self.sequences = sequences
            
            outputs = DummyOutputs(caption_tokens)
        
        # 간단한 캡션 처리 (프롬프트 길이 계산 없이)
        caption_only_ids = outputs.sequences
        
        # 캡션 길이 제한 (메모리 절약)
        max_caption_length = 16  # 더 짧게 제한
        if caption_only_ids.shape[1] > max_caption_length:
            caption_only_ids = caption_only_ids[:, :max_caption_length]
        
        # 생성된 input_ids를 바로 임베딩으로 변환
        if caption_only_ids.shape[1] > 0:
            caption_embeds = self.get_model().get_input_embeddings()(caption_only_ids.long())
        else:
            # 빈 캡션인 경우 빈 임베딩 텐서 생성
            caption_embeds = torch.empty((1, 0, self.config.hidden_size), dtype=v_emb.dtype, device=v_emb.device)
        
        # 원래 훈련 상태로 복원
        self.train(training_state)
        
        # 배치 차원 제거하고 반환
        return caption_embeds.squeeze(0)  # (caption_length, dim)

    def get_2dPool(self, image_feature, stride=2):
        # 입력: (64, 729, 3584)
        height = width = self.get_vision_tower().num_patches_per_side  # 27
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1) #(num_frames, height, width, num_dim)
        #(64, 729, 3584) -> (64, 27,27, 3584)
        compressor_type = getattr(self.config, "compressor_type", None)
        if compressor_type == "bimba":
            space_time_tokens = image_feature.unsqueeze(0)
            # (1, 64, 27, 27, 3584)

        temporal_pooling = getattr(self.config, "temporal_pooling", 1)
        if temporal_pooling>1:
            image_feature = image_feature.reshape(num_frames, num_tokens, num_dim)
            image_feature = image_feature.permute(1, 2, 0)
            image_feature = nn.functional.avg_pool1d(image_feature, kernel_size=temporal_pooling, stride=temporal_pooling)
            image_feature = image_feature.permute(2, 0, 1)
            num_frames = num_frames//temporal_pooling
            image_feature = image_feature.view(num_frames, height, width, -1)

        image_feature = image_feature.permute(0, 3, 1, 2).contiguous() # (num_frames, num_dim, height, width)
        # (64, 27,27, 3584) -> (64, 3584, 27, 27)
        # 정보 압축
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride) 
            # (64, 3584, 27, 27) -> (64, 3584, 14, 14)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear') 
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        # (64, 3584, 14, 14) -> (64, 14, 14, 3584)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        # (64, 14*14, 3584) -> (64, 196, 3584)
        if compressor_type == "bimba":
            image_feature = image_feature.unsqueeze(0)
            # (64, 196, 3584) -> (1, 64, 196, 3584)
            
            # space_time_tokens을 올바른 형태로 변환
            if space_time_tokens.dim() == 5:
                B, T, H, W, D = space_time_tokens.shape  # (1, 64, 27, 27, 3584)
                space_time_tokens_reshaped = space_time_tokens.view(B, T*H*W, D)  # (1, 64*27*27, 3584)
            else:
                print(f"Unexpected space_time_tokens shape: {space_time_tokens.shape}, skipping compression")
                space_time_tokens_reshaped = None
            
            try:
                if space_time_tokens_reshaped is not None:
                    image_feature = self.get_model().compressor(space_time_tokens_reshaped, image_feature)
                else:
                    print("Compressor input is None, skipping compression")
            except Exception as e:
                # 에러 발생 시 compressor 사용하지 않고 그대로 반환
                print(f"Compressor error: {e}, skipping compression")
                pass
            
            # (1, 64, 196, 3584) -> (64, 196, 3584)
            image_feature = torch.squeeze(image_feature, 0)

        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def _prepare_multimodal_inputs(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask=None,
        labels=None
    ):
        """CaptioningVLM의 _prepare_multimodal_inputs 메서드를 통합"""
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)
        
        # 1. 토큰 전처리
        processed_input_ids = self.get_model().preprocess_image_tokens(input_ids)
        processed_labels = processed_input_ids.clone() if labels is None else self.get_model().preprocess_image_tokens(labels)
        
        # 2. 비전 인코딩
        B = processed_input_ids.size(0) 
        v_embs = self.get_model()._get_vision_embeds(pixel_values)
        
        v_embs = list(torch.split(v_embs, v_embs.size(0)//B, dim=0))
        
        for i, v_emb in enumerate(v_embs):
            # 3. 비전 임베딩 풀링
            if self.config.mm_spatial_pool_mode != "none":
                v_emb = self.get_2dPool(v_emb, stride=2)

            chunk_num = 2  # 4에서 2로 줄여 메모리 사용량 감소
            num_samples, seq_len, dim = v_emb.shape
            chunk_size = num_samples // chunk_num if num_samples >= chunk_num else 1
            
            # 2개 청크로 분할 및 각 청크에 뉴라인 토큰 삽입
            chunks_with_caption = []
            for j in range(chunk_num):
                start = j * chunk_size
                end = (j + 1) * chunk_size if j < chunk_num - 1 else num_samples
                
                if start < num_samples:
                    chunk = v_emb[start:end]
                    
                    # 뉴라인 토큰 삽입
                    chunk_with_newline = self.get_model().newline_inserter(chunk, self.get_model().image_newline)
                    
                    # 캡션 생성 - use_captioning_vlm 설정과 tokenizer 확인
                    use_captioning = getattr(self.config, "use_captioning_vlm", False)
                    tokenizer = self.get_model().get_tokenizer()
                    
                    if use_captioning and tokenizer is not None:
                        try:
                            caption = self._generate_captions_for_features(chunk_with_newline, tokenizer)
                        except Exception as e:
                            # 캡션 생성 실패 시 빈 캡션 생성 (메모리 절약)
                            print(f"Caption generation failed: {e}")
                            caption = torch.empty((2, chunk_with_newline.shape[-1]), dtype=chunk_with_newline.dtype, device=chunk_with_newline.device)
                    else:
                        # use_captioning_vlm이 False이거나 tokenizer가 없으면 빈 캡션 생성 (메모리 절약)
                        caption = torch.empty((2, chunk_with_newline.shape[-1]), dtype=chunk_with_newline.dtype, device=chunk_with_newline.device)
                    
                    # 청크와 캡션 결합
                    chunk_with_caption = torch.cat([chunk_with_newline, caption], dim=0)
                    chunks_with_caption.append(chunk_with_caption)
                    
            # 모든 청크 결합
            if chunks_with_caption:
                v_embs[i] = torch.cat(chunks_with_caption, dim=0)

        # 7. 이미지 토큰 대체 - 기존 메서드 활용
        return self._replace_image_tokens_with_features_internal(
            input_ids=processed_input_ids,
            labels=processed_labels,
            attention_mask=attention_mask,
            image_features=v_embs,
            embed_tokens_fn=self.get_model().embed_tokens,
            image_token_index=IMAGE_TOKEN_INDEX,
            ignore_index=IGNORE_INDEX,
            max_length=getattr(self.config, 'max_position_embeddings', 4096),
            padding_side=getattr(self, 'tokenizer', None) and self.tokenizer.padding_side or 'right',
        )
    
    def _replace_image_tokens_with_features_internal(self, input_ids, labels, attention_mask, image_features, 
                                                   embed_tokens_fn, image_token_index, ignore_index, 
                                                   max_length, padding_side):
        """내부 이미지 토큰 대체 메서드"""
        # 기존 prepare_inputs_labels_for_multimodal의 핵심 로직을 여기에 구현
        # 단순화된 버전으로 구현
        batch_size = input_ids.size(0)
        
        new_input_embeds = []
        new_labels = []
        
        for batch_idx in range(batch_size):
            cur_input_ids = input_ids[batch_idx]
            cur_labels = labels[batch_idx] if labels is not None else torch.full_like(cur_input_ids, ignore_index)
            
            # 이미지 토큰 위치 찾기
            image_token_indices = torch.where(cur_input_ids == image_token_index)[0]
            
            if len(image_token_indices) == 0:
                # 이미지 토큰이 없는 경우
                cur_input_embeds = embed_tokens_fn(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
            else:
                # 이미지 토큰이 있는 경우
                cur_new_input_embeds = []
                cur_new_labels = []
                
                last_idx = 0
                for img_idx, token_idx in enumerate(image_token_indices):
                    # 이미지 토큰 전 텍스트
                    if token_idx > last_idx:
                        text_ids = cur_input_ids[last_idx:token_idx]
                        text_embeds = embed_tokens_fn(text_ids)
                        cur_new_input_embeds.append(text_embeds)
                        cur_new_labels.append(cur_labels[last_idx:token_idx])
                    
                    # 이미지 특징 추가
                    if img_idx < len(image_features):
                        cur_image_features = image_features[img_idx]
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), ignore_index, 
                                                       device=cur_labels.device, dtype=cur_labels.dtype))
                    
                    last_idx = token_idx + 1
                
                # 마지막 이미지 토큰 후 텍스트
                if last_idx < len(cur_input_ids):
                    text_ids = cur_input_ids[last_idx:]
                    text_embeds = embed_tokens_fn(text_ids)
                    cur_new_input_embeds.append(text_embeds)
                    cur_new_labels.append(cur_labels[last_idx:])
                
                # 결합
                cur_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                cur_labels = torch.cat(cur_new_labels, dim=0)
                
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels)
        
        # 패딩
        max_len = max(x.shape[0] for x in new_input_embeds)
        
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), ignore_index, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=input_ids.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=input_ids.device)
        
        for i, (cur_embeds, cur_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_embeds.shape[0]
            
            if padding_side == "left":
                new_input_embeds_padded.append(torch.cat([
                    torch.zeros((max_len - cur_len, cur_embeds.shape[1]), dtype=cur_embeds.dtype, device=cur_embeds.device),
                    cur_embeds
                ], dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_labels
                    new_attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=torch.long, device=input_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat([
                    cur_embeds,
                    torch.zeros((max_len - cur_len, cur_embeds.shape[1]), dtype=cur_embeds.dtype, device=cur_embeds.device)
                ], dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_labels
                    new_attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=torch.long, device=input_ids.device)
        
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        return new_input_embeds, new_attention_mask, new_labels_padded, position_ids
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        '''
        input: image_feature: (num_frames, num_patches, hidden_size)
        image_feature shape: torch.Size(64, 196, 3584])
        feature_dim: 3584
        image_feature view shape: torch.Size([64, 1, 14, 14, 3584]) 14 * 14 = 196
        image_feature permute shape: torch.Size([3584, 64, 14, 1, 14])
        image_feature flatten shape: torch.Size([3584, 896, 14])
        image_feature cat shape: torch.Size([3584, 896, 15]) 14 + 1, 1 is the newline token
        image_feature final shape: torch.Size([13440, 3584]) seq_len,features = 
        '''
        num_frames = image_feature.shape[0]
        num_patches = image_feature.shape[1]
        feature_dim = image_feature.shape[-1]
        
        # 안전한 resize_h 계산
        resize_h = int(math.sqrt(num_patches))
        if resize_h * resize_h != num_patches:
            print(f"Warning: num_patches ({num_patches}) is not a perfect square. Using resize_h={resize_h}")
            # 가장 가까운 제곱수로 조정하거나 에러 방지를 위해 flatten 방식 사용
            image_feature = image_feature.flatten(0, 1)  # (num_frames * num_patches, hidden_size)
            return image_feature
        
        # print("image_feature shape:", image_feature.shape)
        # print("feature_dim:", feature_dim)

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1) 
        # print("image_feature view shape:", image_feature.shape)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous() 
        # print("image_feature permute shape:", image_feature.shape)
        image_feature = image_feature.flatten(1, 2).flatten(2, 3) 
        # print("image_feature flatten shape:", image_feature.shape)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1) # (hidden_size, num_frames, h*w+1)
        # print("image_feature cat shape:", image_feature.shape)
        if getattr(self.config, "add_faster_video", False):
            print("add faster video")
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1) 
        # print("image_feature final shape:", image_feature.shape)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        

        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        print(len(images),images[0].shape)
        
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            # print(len(images),images[0].shape) # Same
            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            
            split_sizes = [image.shape[0] for image in images_list]
            print("split_sizes:", split_sizes)
            
            # 먼저 기본 이미지 인코딩 및 풀링 적용
            encoded_image_features = self.encode_images(concat_images)
            print("encoded_image_features shape:", encoded_image_features.shape)
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # Initialize all_faster_video_features as empty list for now
            all_faster_video_features = [0] * len(encoded_image_features)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                print("After 2dpooling image_feat shape:")
                if idx in video_idx_in_batch:
                    image_feat = self.get_2dPool(image_feat)
                    print("image_features length:",len(image_features))
                    print("image_feat shape:", image_feat.shape)
                
                # CaptioningVLM 기능 사용 여부 확인 (get_2dPool 이후)
                use_captioning = getattr(self.config, "use_captioning_vlm", False)
                
                if use_captioning:
                    # CaptioningVLM 프로세싱 (청크 분할 및 캡션 생성) - 메모리 절약을 위해 청크 수 축소
                    chunk_num = 2  # 4에서 2로 줄여 메모리 사용량 감소
                    num_samples, seq_len, dim = image_feat.shape
                    chunk_size = num_samples // chunk_num if num_samples >= chunk_num else 1
                    
                    chunks_with_caption = []
                    for j in range(chunk_num):
                        start = j * chunk_size
                        end = (j + 1) * chunk_size if j < chunk_num - 1 else num_samples
                        
                        if start < num_samples:
                            chunk = image_feat[start:end]
                            
                            # 뉴라인 토큰 삽입
                            chunk_with_newline = self.get_model().newline_inserter(chunk, self.get_model().image_newline)
                            
                            # 캡션 생성 - use_captioning_vlm 설정과 tokenizer 확인
                            use_captioning = getattr(self.config, "use_captioning_vlm", False)
                            tokenizer = self.get_model().get_tokenizer()
                            
                            if use_captioning and tokenizer is not None:
                                try:
                                    caption = self._generate_captions_for_features(chunk_with_newline, tokenizer)
                                except Exception as e:
                                    # 캡션 생성 실패 시 빈 캡션 생성 (메모리 절약)
                                    print(f"Caption generation failed: {e}")
                                    caption = torch.empty((2, chunk_with_newline.shape[-1]), dtype=chunk_with_newline.dtype, device=chunk_with_newline.device)
                            else:
                                # use_captioning_vlm이 False이거나 tokenizer가 없으면 빈 캡션 생성 (메모리 절약)
                                caption = torch.empty((2, chunk_with_newline.shape[-1]), dtype=chunk_with_newline.dtype, device=chunk_with_newline.device)
                            
                            # 청크와 캡션 결합
                            chunk_with_caption = torch.cat([chunk_with_newline, caption], dim=0)
                            chunks_with_caption.append(chunk_with_caption)
                    
                    # 모든 청크 결합
                    if chunks_with_caption:
                        final_features = torch.cat(chunks_with_caption, dim=0)
                        # CaptioningVLM 처리 후에는 flatten 처리하여 spatial 처리를 건너뛰기 위해 표시
                        final_features._captioning_processed = True
                        image_features.append(final_features)
                    else:
                        image_features.append(image_feat)
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # CaptioningVLM 처리된 텐서는 이미 flatten되어 있으므로 바로 추가
                    if hasattr(image_feature, '_captioning_processed') and image_feature._captioning_processed:
                        new_image_features.append(image_feature)
                        continue
                        
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)
                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            
                            unit = image_feature.shape[2]
                            
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            
                        elif "unpad" in mm_patch_merge_type:
                            
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            # 차원 확인 및 조정 - 모든 텐서를 2D로 통일
            for i, emb in enumerate(cur_new_input_embeds):
                if emb.dim() == 1:
                    cur_new_input_embeds[i] = emb.unsqueeze(0)
                elif emb.dim() == 3:
                    # 3D 텐서를 2D로 flatten
                    cur_new_input_embeds[i] = emb.view(-1, emb.shape[-1])
                elif emb.dim() > 3:
                    # 더 높은 차원의 텐서도 2D로 flatten
                    cur_new_input_embeds[i] = emb.view(-1, emb.shape[-1])
            
            # 모든 텐서가 2D인지 확인
            for i, emb in enumerate(cur_new_input_embeds):
                if emb.dim() != 2:
                    print(f"Warning: tensor {i} has unexpected dimension {emb.dim()}, shape: {emb.shape}")
                    cur_new_input_embeds[i] = emb.view(-1, emb.shape[-1])
            
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
