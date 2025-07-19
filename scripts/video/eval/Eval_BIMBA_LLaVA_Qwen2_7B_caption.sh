export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
export PYTHONPATH="$PYTHONPATH:/home/tera/workspace/lcvlm/bimba_fork/llava"

model_path="checkpoints/BIMBA-LLaVA-Qwen2-7B"
model_base="lmms-lab/LLaVA-Video-7B-Qwen2"
model_name="withcaption"

results_dir=results/BIMBA-LLaVA-Qwen2-7B_caption

dataset_name=NextQA
python llava/eval/infer.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_name $model_name \
    --results_dir ${results_dir}/${dataset_name}_test \
    --max_frames_num 64 \
    --dataset_name $dataset_name \
    --data_path DATAS/eval/NextQA/formatted_test.json \
    --video_root "DATAS/eval/NextQA/NExTVideo/" \
    --cals_acc

# python llava/eval/submit_ego_schema.py
# kaggle competitions submit -c egoschema-public -f results/BIMBA-LLaVA-Qwen2-7B/EgoSchema/es_submission.csv -m "BIMBA-LLaVA-Qwen2-7B"

