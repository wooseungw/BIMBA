import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
import av
import numpy as np
from decord import VideoReader, cpu


def load_video(video_path, max_frames_num, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0.0

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    # 전체 재생시간 (초)
    video_time = float(stream.duration * stream.time_base)

    keyframes = []
    timestamps = []
    for packet in container.demux(stream):
        if not packet.is_keyframe:
            continue
        for frame in packet.decode():
            img = frame.to_ndarray(format="rgb24")
            keyframes.append(img)
            # Fraction → float으로 변환
            time_sec = float(frame.pts * stream.time_base)
            timestamps.append(time_sec)
        if len(keyframes) >= max_frames_num:
            break

    if force_sample and len(keyframes) > max_frames_num:
        idxs = np.linspace(0, len(keyframes)-1, max_frames_num, dtype=int)
        keyframes = [keyframes[i] for i in idxs]
        timestamps = [timestamps[i] for i in idxs]
    print(len(keyframes),keyframes[0].shape)  # 이제 float이므로 포맷 가능
    print(timestamps)
    # 이제 float이므로 포맷 가능
    frame_time = ",".join([f"{t:.2f}s" for t in timestamps])
    frames = np.stack(keyframes, axis=0)

    return frames, frame_time, video_time

model_path = "checkpoints/BIMBA-LLaVA-Qwen2-7B"
model_base = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen_lora"


device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(
                                                    model_path = model_path, 
                                                    model_base = model_base, 
                                                    model_name = model_name, 
                                                    torch_dtype="bfloat16", 
                                                    device_map=device_map,
                                                    attn_implementation=None,
                                                )

model.eval()


video_path = "assets/example.mp4"
max_frames_num = 64
video,frame_time,video_time = load_video(video_path, max_frames_num, force_sample=True)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
video = [video]
conv_template = "qwen_1_5"
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