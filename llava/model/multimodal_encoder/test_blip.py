# test_blip2_tower.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# test_blip2_tower.py
import os
import sys

# 프로젝트 루트 경로를 시스템 경로에 추가
sys.path.append('e://Workspace/BIMBA')
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
        vision_tower="Salesforce/blip2-opt-2.7b", 
    )

    # 3) forward 호출
    with torch.no_grad():
        features = tower([test_img],[texts])

    # 4) 결과 출력
    for feature  in features:
        print(feature["image_features"].shape)  # [1, 36, 1024]
        print(feature["captions"])
        

if __name__ == "__main__":
    test_blip2_vision_tower()