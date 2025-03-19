import argparse
import copy
import torch
import warnings
from decord import VideoReader, cpu
import numpy as np
import json
import multiprocessing as mp
import os
from multiprocessing import Pool
import functools
import itertools
import random
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import rank0_print


warnings.filterwarnings("ignore")

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()


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

def get_options_letter(len_options):
    if len_options==2:
        return '(A or B)'
    elif len_options==3:
        return '(A, B or C)'
    elif len_options==4:
        return '(A, B, C or D)'
    elif len_options==5:
        return '(A, B, C, D, or E)'
    else:
        raise NotImplementedError

def get_prompt(dataset_name, sample, conv_template="qwen_1_5", video_time=None, num_frames=None, frame_time=None):
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    if video_time:
        prompt = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}.\n"
    else:
        prompt = ""

    if dataset_name in ['VSI']:
        prompt += "These are frames of a video.\n"
        prompt += sample["question"] + "\n"
        if 'candidates' in sample:
            for op in sample["candidates"]:
                prompt += f"{op}\n"
            prompt += "Answer with the option's letter from the given choices directly."
        else:
            prompt += "Please answer the question using a single word or phrase."
    elif dataset_name in ['MovieChat']:
        if video_time is None:
            prompt += "These are frames of a video.\n"
        if 'time' in sample:
            timestamp = round(sample['time']/sample['fps'], 2)
            prompt += f"At time {timestamp}s, "
        prompt += sample["question"] + "\n"
        prompt += "Please answer the question using a single word, phrase, or sentence."
        #prompt += "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
    else:
        options_letter = get_options_letter(len(sample['candidates']))
        prompt += f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter {options_letter} of the correct option.\n"
        prompt += sample["question"] + "\n"
        for op in sample["candidates"]:
            prompt += f"{op}\n"
        prompt += f"The best answer is:"
        
    question = DEFAULT_IMAGE_TOKEN + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def run(rank, world_size, args):
    torch.cuda.set_device(rank)

    rank0_print("Loadind dataset from", args.data_path)
    with open(args.data_path, "r") as f:
        dataset = json.load(f)
     
    random.shuffle(dataset)

    num_samples = int(len(dataset) * args.test_ratio)
    dataset = dataset[rank:num_samples:world_size]
    rank0_print(f"Total samples: {num_samples}")
    print(f"Samples in rank {rank}: {len(dataset)}")

    tokenizer, model, image_processor, max_length = load_pretrained_model(
                                                        model_path = args.model_path, 
                                                        model_base = args.model_base, 
                                                        model_name = args.model_name,
                                                        lora_alpha = args.lora_alpha,
                                                        torch_dtype="bfloat16",
                                                        device_map="auto",
                                                        #device_map = {"": torch.device(f"cuda:{rank}")},
                                                    )
    model.eval()
    #model = model.to(torch.device(rank))


    result_list = []
    for cnt, sample in enumerate(tqdm(dataset)):
        sample_save_path = f"{args.results_dir}/outputs/{sample['id']}.json"
        if os.path.exists(sample_save_path):
            with open(sample_save_path, 'r') as f:
                sample = json.load(f)
        else:
            video_path = os.path.join(args.video_root, sample["video"])
            video,frame_time,video_time = load_video(video_path, args.max_frames_num, fps=1, force_sample=True)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            video = [video]
            if args.use_time_ins:
                prompt_question = get_prompt(args.dataset_name, sample, video_time=video_time, num_frames=args.max_frames_num, frame_time=frame_time)
            else:
                prompt_question = get_prompt(args.dataset_name, sample)

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            cont = model.generate(
                input_ids,
                images=video,
                modalities= ["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new_tokens,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            sample["prediction"] = text_outputs

            with open(sample_save_path, "w") as f:
                json.dump(sample, f, indent=4)
        
        result_list.append(sample)
        if "answer" in sample:
            print(cnt, "GT:", sample["answer"], "Pred:", sample["prediction"])
        else:
            print(cnt, "Pred:", sample["prediction"])
    
    return result_list


def main():
    parser = argparse.ArgumentParser(description="Run Inference")

    # Model
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--max_frames_num", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--conv_template", type=str, default="qwen_1_5")
    parser.add_argument("--use_time_ins", action="store_true")
    parser.add_argument("--lora_alpha", type=int, default=None)

    # Data
    parser.add_argument("--dataset_name", type=str, default="VideoMME")
    parser.add_argument("--data_path", type=str, default="/mnt/bum/mmiemon/datasets/Video-MME/formatted_dataset.json")
    parser.add_argument("--video_root", type=str, default="/mnt/bum/mmiemon/datasets/Video-MME/videos/data")
    parser.add_argument("--results_dir", type=str, default="/mnt/bum/mmiemon/LLaVA-NeXT/results/llava_video/VideoMME")
    parser.add_argument("--test_ratio", type=float, default=1)
    parser.add_argument("--multiprocess", action="store_true")
    parser.add_argument("--cals_acc", action="store_true")

    args = parser.parse_args()
    if args.model_base == "None":
        args.model_base = None

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/outputs", exist_ok=True)

    if args.multiprocess:
        mp.set_start_method("spawn")
        print(f"started benchmarking")
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        print("World size", world_size)
        with Pool(world_size) as pool:
            func = functools.partial(run, args=args, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        print("finished running")
        result_list = [res for res in itertools.chain(*result_lists)]
    else:
        result_list = run(0, world_size=1, args=args)
    

    if args.cals_acc:
        results = {"all": {"correct": 0, "total": 0}}
        for sample in result_list:
            if "answer" not in sample:
                continue
            results["all"]["total"] += 1
            if "question_type" in sample:
                if sample["question_type"] not in results:
                    results[sample["question_type"]] = {"correct": 0, "total": 0}
                results[sample["question_type"]]["total"] += 1
                
            if sample["answer"].lower()==fuzzy_matching(sample["prediction"]).lower():
                results["all"]["correct"] += 1
                if "question_type" in sample:
                    results[sample["question_type"]]["correct"] += 1

        for key in results:
            results[key]["accuracy"] = results[key]["correct"] / results[key]["total"]

        print(results)

        with open(os.path.join(args.results_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()