#!/bin/zsh

export data_root="/gemini/platform/public/embodiedAI/users/ysy/data"
export code_root="/gemini/space/users/ysy/project/VLA-tts"

export TOKENIZERS_PARALLELISM="false"

DATASET_REPO_ID="RoboTwin/all_tasks_50ep"
# DATASET_REPO_ID="RoboTwin/eval_cfn/15tasks_25epis"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"
# DATASET_ROOT="/gemini/space/users/ysy/data/dataset/lerobot_robotwin_dataset"

# Output directory
# OUTPUT_DIR="/gemini/space/users/ysy/data/train_cfn/trans-0801"
rm -r "${data_root}/train_cfn/temp"
OUTPUT_DIR="${data_root}/train_cfn/temp"

# Training Parameters
BATCH_SIZE=64 # 64 # 32
# TOTAL_STEPS=280000
SAVE_FREQ=1
ACTION_CHUNK_SIZE=30
NUM_WORKERS=2

gpu_id=0
export CUDA_VISIBLE_DEVICES=${gpu_id}

cd ${code_root}/scripts/eval/eval_cnn_cfn
# kernprof -l -v train2.py \
python eval_cfn_cnn_prior.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --dataset.image_transforms.enable=true \
    --dataset.wrist_transforms.enable=true \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
    --policy.type='pi0' --policy.use_delta_action=false \
    # --policy.chunk_size=$ACTION_CHUNK_SIZE --policy.n_action_steps=$ACTION_CHUNK_SIZE

