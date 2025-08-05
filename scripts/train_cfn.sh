#!/bin/zsh

export TOKENIZERS_PARALLELISM="false"

DATASET_REPO_ID="RoboTwin/all_tasks_50ep"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"
# DATASET_ROOT="/gemini/space/users/ysy/data/dataset/lerobot_robotwin_dataset"

# Output directory
OUTPUT_DIR="/gemini/space/users/ysy/data/train_cfn/mlp-0723"
# rm -r $OUTPUT_DIR

# Training Parameters
BATCH_SIZE=64
# TOTAL_STEPS=280000
SAVE_FREQ=1
ACTION_CHUNK_SIZE=50
NUM_WORKERS=8

cd /gemini/space/users/ysy/project/vla_post_train/
python train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --dataset.image_transforms.enable=true \
    --dataset.wrist_transforms.enable=true \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
    --policy.type='pi0' --policy.use_delta_action=false \
    # --policy.chunk_size=$ACTION_CHUNK_SIZE --policy.n_action_steps=$ACTION_CHUNK_SIZE
