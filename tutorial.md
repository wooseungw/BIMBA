# 학습
```bash
bash scripts/video/train/Train_BIMBA_LLaVA_Qwen2_7B_caption.sh
```

# 검증
```bash
bash scripts/video/eval/Eval_BIMBA_LLaVA_Qwen2_7B_caption.sh
```
학습sh파일은 복사해서 run name을 바꿔서 사용합니다.(아니면 덮어쓰기 됨)
# 학습시 캡션 생성 인자 수정

1. Train.py 파일에서 `caption` 인자를 `True`로 설정합니다.
2. captioning_system_instruction 으로 캡션 생성 시스템 지시어를 설정합니다.
3. captioning_instruction 으로 캡션 생성 프롬프트를 설정합니다.


