

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


dataset_name=MLVU
python llava/eval/infer.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_name $model_name \
    --results_dir ${results_dir}/${dataset_name} \
    --max_frames_num 64 \
    --dataset_name $dataset_name \
    --data_path DATAS/eval/MLVU/formatted_dataset.json \
    --video_root "path_to_video_folder" \
    --cals_acc

dataset_name=LongVideoBench
python llava/eval/infer.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_name $model_name \
    --results_dir ${results_dir}/${dataset_name}_val \
    --max_frames_num 64 \
    --dataset_name $dataset_name \
    --data_path DATAS/eval/LongVideoBench/formatted_dataset.json \
    --video_root "path_to_video_folder" \
    --cals_acc

dataset_name=NextQA
python llava/eval/infer.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_name $model_name \
    --results_dir ${results_dir}/${dataset_name}_test \
    --max_frames_num 64 \
    --dataset_name $dataset_name \
    --data_path DATAS/eval/NextQA/formatted_dataset_test.json \
    --video_root "path_to_video_folder" \
    --cals_acc

dataset_name=PerceptionTest
python llava/eval/infer.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_name $model_name \
    --results_dir ${results_dir}/${dataset_name} \
    --max_frames_num $max_frames_num \
    --dataset_name $dataset_name \
    --data_path DATAS/eval/PerceptionTest/formatted_dataset_val.json \
    --video_root "path_to_video_folder" \
    --cals_acc

dataset_name=VNBench
python llava/eval/infer.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_name $model_name \
    --results_dir ${results_dir}/${dataset_name} \
    --max_frames_num 128 \
    --dataset_name $dataset_name \
    --data_path DATAS/eval/VNBench/formatted_dataset_4try.json \
    --video_root "path_to_video_folder" \
    --cals_acc

#EgoSchema subset
dataset_name=EgoSchema
python llava/eval/infer.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_name $model_name \
    --results_dir ${results_dir}/${dataset_name} \
    --max_frames_num 64 \
    --dataset_name $dataset_name \
    --data_path DATAS/eval/EgoSchema/formatted_dataset_val.json \
    --video_root "path_to_video_folder" \
    --cals_acc

#EgoSchema full-set
dataset_name=EgoSchema
python llava/eval/infer.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_name $model_name \
    --results_dir ${results_dir}/${dataset_name} \
    --max_frames_num 64 \
    --dataset_name $dataset_name \
    --data_path DATAS/eval/EgoSchema/formatted_dataset_test.json \
    --video_root "path_to_video_folder"


# python llava/eval/submit_ego_schema.py
# kaggle competitions submit -c egoschema-public -f results/BIMBA-LLaVA-Qwen2-7B/EgoSchema/es_submission.csv -m "BIMBA-LLaVA-Qwen2-7B"

