#!/bin/zsh
export data_root="/gemini/platform/public/embodiedAI/users/ysy/data"
export code_root="/gemini/space/users/ysy/project/vla_post_train"

export TOKENIZERS_PARALLELISM="false"

DATASET_ROOT="/gemini/platform/public/embodiedAI/shared_dataset/Realman/WAIC/lerobot_pure_env/with_s6f/only_lid"

# Output directory
# OUTPUT_DIR="/gemini/space/users/ysy/data/train_cfn/trans-0801"
OUTPUT_DIR="${data_root}/train_cfn/rm_only_lid2-0805"
# rm -r "${data_root}/train_cfn/temp"

# Training Parameters
BATCH_SIZE=64 # 64 # 32
# TOTAL_STEPS=280000
SAVE_FREQ=1
ACTION_CHUNK_SIZE=20
NUM_WORKERS=16

gpu_id=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

cd $code_root
# kernprof -l -v train2.py \
python train2.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --dataset.image_transforms.enable=true \
    --dataset.wrist_transforms.enable=true \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
    --policy.type='pi0' --policy.use_delta_action=false \
    # --policy.chunk_size=$ACTION_CHUNK_SIZE --policy.n_action_steps=$ACTION_CHUNK_SIZE
