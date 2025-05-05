# import torch
# from transformers import CLIPModel, CLIPProcessor
# from tqdm import tqdm     # 프로그래스 바용
# from PIL import Image
# import math
# import requests

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_id  = "openai/clip-vit-base-patch32"
# model     = CLIPModel.from_pretrained(model_id).to(device).eval()
# processor = CLIPProcessor.from_pretrained(model_id)

# # 1) 이미지 피쳐
# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# response = requests.get(img_url, stream=True)
# raw_image = Image.open(response.raw).convert('RGB')
# img_inputs= processor(images=raw_image, return_tensors="pt").to(device)
# with torch.no_grad():
#     img_feat = model.get_image_features(**img_inputs)       # (1, D)
# img_feat    = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)

# # 2) 어휘 전체 토큰 리스트 준비 (특수 토큰은 제외)
# tokenizer = processor.tokenizer
# all_ids   = list(range(tokenizer.vocab_size))
# # 예: special tokens 또는 너무 짧은 토큰은 필터링
# valid_ids = [i for i in all_ids
#              if tokenizer.convert_ids_to_tokens(i).strip() not in tokenizer.all_special_tokens]

# # 3) 배치 단위로 텍스트 피쳐 계산
# batch_size = 512  # 메모리 여건에 따라 조절
# text_feats = []

# for i in tqdm(range(0, len(valid_ids), batch_size), desc="Computing text features"):
#     batch_ids = valid_ids[i : i + batch_size]
#     tokens    = tokenizer.convert_ids_to_tokens(batch_ids)
#     enc       = tokenizer(tokens,
#                           return_tensors="pt",
#                           padding=True,
#                           truncation=True,
#                           max_length=1).to(device)
#     with torch.no_grad():
#         feats = model.get_text_features(**enc)             # (B, D)
#     feats     = feats / feats.norm(p=2, dim=-1, keepdim=True)
#     text_feats.append(feats.cpu())

# text_feats = torch.cat(text_feats, dim=0)  # (V', D)

# # 4) 코사인 유사도 계산 & Top-K
# #    img_feat: (1, D), text_feats.T: (D, V') → sims: (1, V')
# sims   = img_feat.cpu() @ text_feats.T      # (1, V')
# K      = 10
# values, indices = sims[0].topk(K)

# # 5) 결과 디코딩
# for idx, score in zip(indices.tolist(), values.tolist()):
#     tok = tokenizer.convert_ids_to_tokens(valid_ids[idx])
#     print(f"{tok}\t{score:.4f}")