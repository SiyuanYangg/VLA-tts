#!/bin/zsh

# Dataset configuration
# DATASET_REPO_ID="lerobot/aloha_mobile_cabinet"
# DATASET_ROOT="/data/zhangyang/huggingface_cache/hub/datasets--lerobot--aloha_mobile_cabinet"

# DATASET_REPO_ID="RoboMind/tienkung_gello_1rgb_normkey"
DATASET_REPO_ID="RoboTwin/all_tasks_50ep"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"
DATASET_ROOT="/gemini/space/users/ysy/data/dataset/lerobot_robotwin_dataset"

# Output directory
OUTPUT_DIR="/gemini/space/users/ysy/data/train_cfn/temp"
rm -r $OUTPUT_DIR

# Training Parameters
BATCH_SIZE=480
# TOTAL_STEPS=280000
SAVE_FREQ=1
ACTION_CHUNK_SIZE=50
NUM_WORKERS=16

cd /gemini/space/users/ysy/project/vla_post_train/
python eval.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --dataset.image_transforms.enable=true \
    --dataset.wrist_transforms.enable=true \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
    --policy.type='pi0' --policy.use_delta_action=false
    # --policy.chunk_size=$ACTION_CHUNK_SIZE --policy.n_action_steps=$ACTION_CHUNK_SIZE
