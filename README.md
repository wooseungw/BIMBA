# Vamba

This repo contains code for [Vamba](https://arxiv.org/abs/TODO), a hybrid Mamba-Transformer model that leverages cross-attention layers and Mamba-2 blocks for efficient hour-long video understanding.

[**🌐 Homepage**](https://sites.google.com/view/bimba-mllm) | [**📖 arXiv**](https://arxiv.org/abs/2503.09590) | [**💻 GitHub**]() | [**🤗 Model**](BIMBA-LLaVA-NeXT/checkpoints/BIMBA-LLaVA-Qwen2-7B)

## Install
Please use the following commands to install the required packages:
```bash
conda create --name bimba python=3.10
conda activate bimba
pip install -r requirements.txt
```
This codebase is built on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and [mamba](https://github.com/state-spaces/mamba) codebases.
## Model Inference
```python
# git clone https://github.com/TIGER-AI-Lab/Vamba
# cd BIMBA-LLaVA-NeXT
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

model_path = "checkpoints/BIMBA-LLaVA-Qwen2-7B"
model_base = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen_lora"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(model_path = model_path, model_base = model_base, model_name = model_name, torch_dtype="bfloat16", device_map=device_map,attn_implementation=None)
model.eval()

video_path = "/mnt/opr/ce/datasets/NextQA/videos/0000/2440175990.mp4"
max_frames_num = 64
video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
video = [video]
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\nPlease describe this video in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
cont = model.generate(
    input_ids,
    images=video,
    modalities= ["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)
```

## Model Training
1. Follow the [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_Video_1003.md) codebase to prepare the training data (e.g., [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K)).
Update the [exp.yaml](BIMBA-LLaVA-NeXT/scripts/video/train/exp.yaml) file to point to your data.
2. Follow the commands below to train BIMBA model:
```bash
cd BIMBA-LLaVA-NeXT
bash scripts/video/train/Train_BIMBA_LLaVA_Qwen2_7B.sh
```

## Evaluation

### Evaluate A Single Dataset (e.g., [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME))

1. First, download the videos from the huggingface/dataset repo and replace "path_to_video_folder" accordingly.
2. We provide the formatted json files for the evaluation datasets in the `BIMBA-LLaVA-NeXT/DATAS/eval` folder. You can format a new dataset using the script.  
```python
cd BIMBA-LLaVA-NeXT
python llava/eval/format_eval_data.py
```
3. Use the following script to evaluate a particular dataset.
```bash
cd BIMBA-LLaVA-NeXT
model_path = "checkpoints/BIMBA-LLaVA-Qwen2-7B"
model_base = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen_lora"
results_dir=results/BIMBA-LLaVA-Qwen2-7B
dataset_name=VideoMME
python llava/eval/infer.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_name $model_name \
    --results_dir ${results_dir}/${dataset_name} \
    --max_frames_num 64 \
    --dataset_name $dataset_name \
    --video_root "path_to_video_folder" \
    --data_path DATAS/eval/VideoMME/formatted_dataset.json \
    --cals_acc
```

### Evaluate All Benchmarks

1. Use the following script to evaluate PerceptionTest, NextQA, EgoSchema, VNBench, LongVideoBench, VideoMME, and MLVU  benchmarks.
```bash
cd BIMBA-LLaVA-NeXT
bash scripts/video/eval/Eval_BIMBA_LLaVA_Qwen2_7B.sh
```

2. For EgoSchema, use the following script to prepare the submission file.
```python
python llava/eval/submit_ego_schema.py
```
Then, you can either submit directly to the [kaggle competition page](https://www.kaggle.com/competitions/egoschema-public/overview) or use the script for submission and evaluation.
```
kaggle competitions submit -c egoschema-public -f results/BIMBA-LLaVA-Qwen2-7B/EgoSchema/es_submission.csv -m "BIMBA-LLaVA-Qwen2-7B"
```

3. You should get the following results on these benchmarks.

<!-- |               | EgoSchema | VNBench | VideoMME | MLVU | LongVideoBench | NextQA | PerceptionTest |
|:------------:|:--------:|:------:|:--------:|:----:|:--------------:|:------:|:--------------:|
| **Results**  |   71.14   |  77.88  |   64.67   | 71.37 |     59.46      |  83.73  |      68.51      | -->

<table style="border: 1px solid black; border-collapse: collapse;">
  <tr>
    <th style="border: 1px solid black; padding: 8px;">Dataset</th>
    <th style="border: 1px solid black; padding: 8px;">EgoSchema</th>
    <th style="border: 1px solid black; padding: 8px;">VNBench</th>
    <th style="border: 1px solid black; padding: 8px;">VideoMME</th>
    <th style="border: 1px solid black; padding: 8px;">MLVU</th>
    <th style="border: 1px solid black; padding: 8px;">LongVideoBench</th>
    <th style="border: 1px solid black; padding: 8px;">NextQA</th>
    <th style="border: 1px solid black; padding: 8px;">PerceptionTest</th>
  </tr>
  <tr>
    <th style="border: 1px solid black; padding: 8px;">Results</th>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">71.14</td>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">77.88</td>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">64.67</td>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">71.37</td>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">59.46</td>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">83.73</td>
    <td style="border: 1px solid black; padding: 8px; text-align: center;">68.51</td>
  </tr>
</table>




## Citation
If you find BIMBA useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@article{islam2025bimba,
  title={BIMBA: Selective-Scan Compression for Long-Range Video Question Answering},
  author={Islam, Md Mohaiminul and Nagarajan, Tushar and Wang, Huiyu and Bertasius, Gedas and Torresani, Lorenzo},
  journal={arXiv preprint arXiv:2503.09590},
  year={2025}
}
```