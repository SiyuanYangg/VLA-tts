#!/bin/zsh

export data_root="/gemini/platform/public/embodiedAI/users/ysy/data"
export code_root="/gemini/space/users/ysy/project/VLA-tts"
export TOKENIZERS_PARALLELISM="false"

gpu_id=0
export CUDA_VISIBLE_DEVICES=${gpu_id}

# all
    # block_hammer_beat  block_handover     blocks_stack_easy     blocks_stack_hard   \
    # bottle_adjust      container_place    diverse_bottles_pick  dual_bottles_pick_easy  dual_bottles_pick_hard \
    # dual_shoes_place   empty_cup_place    mug_hanging_easy      mug_hanging_hard \
    # pick_apple_messy   put_apple_cabinet  shoe_place            tool_adjust

    # block_hammer_beat        \
    # bottle_adjust      dual_bottles_pick_hard \
    # dual_shoes_place   empty_cup_place     \
    # pick_apple_messy   put_apple_cabinet  
    # shoe_place            tool_adjust \
    # blocks_stack_easy    blocks_stack_hard \
    # mug_hanging_easy    mug_hanging_hard
# done
    # block_handover  container_place diverse_bottles_pick  dual_bottles_pick_easy 

for task in \
    mug_hanging_easy    mug_hanging_hard        
do
    echo now task = ${task} !!!

    DATASET_REPO_ID="RoboTwin/eval_cfn/single_task/${task}_50epis"
    DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"

    # Output directory
    OUTPUT_DIR="${data_root}/dataset/feature_rt/${task}"
    # rm -r "${data_root}/train_cfn/temp"
    # OUTPUT_DIR="${data_root}/train_cfn/temp"
        
    # Training Parameters
    BATCH_SIZE=128
    # TOTAL_STEPS=280000
    SAVE_FREQ=4
    ACTION_CHUNK_SIZE=30
        NUM_WORKERS=16

    cd ${code_root}/ex/collect
    python co.py \
        --dataset.repo_id=$DATASET_REPO_ID \
        --dataset.root=$DATASET_ROOT \
        --dataset.image_transforms.enable=true \
        --dataset.wrist_transforms.enable=true \
        --output_dir=$OUTPUT_DIR \
        --batch_size=$BATCH_SIZE --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
        --policy.type='pi0' --policy.use_delta_action=true \
        # --policy.chunk_size=$ACTION_CHUNK_SIZE --policy.n_action_steps=$ACTION_CHUNK_SIZE

done

# zsh /gemini/space/users/ysy/000/0-1.sh