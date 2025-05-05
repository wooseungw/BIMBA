# test_blip2_tower.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import requests
from PIL import Image
import torch
from blip_encoder import Blip2VisionTower
from torchvision import transforms

def test_blip2_vision_tower():
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    response = requests.get(img_url, stream=True)
    raw_image = Image.open(response.raw).convert('RGB')
    
    # Convert PIL image to tensor
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])
    test_img = transform(raw_image)
    texts = "Extract Object-Action pair."

    # 2) 모델 초기화
    tower = Blip2VisionTower(
        model_name="Salesforce/blip2-opt-2.7b", 
        vision_tower_cfg=None,
    )
    tower.eval()

    # 3) forward 호출
    with torch.no_grad():
        features = tower([test_img],[texts])

    # 4) 결과 출력
    print(f"■ features.shape: ",features["image_features"].shape)   # (1, seq_len, hidden_size)
    print(f"■ captions     : {features['captions']}")         # ["...생성된 캡션..."]

if __name__ == "__main__":
    test_blip2_vision_tower()