import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from transformers import set_seed
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from cfn.pi0_cfn.cfn_net_pi_prior import CFNWrapper_pi_prior
from cfn.cfn_dataset import cfn_lerobot_dataset

from lerobot.common.datasets.lerobot_dataset import (
    # LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.factory import resolve_delta_timestamps
import json


def make_dataset(cfg: TrainPipelineConfig) -> cfn_lerobot_dataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    # image_transforms = (
    #     ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    # )
    
    if cfg.dataset.repo_id.startswith("["):
        cfg.dataset.repo_id = eval(cfg.dataset.repo_id)

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        # 获取observation.images开头的第一个key
        image_key = next(
            (key for key in ds_meta.features.keys() if key.startswith("observation.images.")),
            None
        )
        if image_key is None:
            raise ValueError("No image key found in the dataset")
        
        # 获取图像维度名称列表
        image_dim_names = ds_meta.features[image_key]['names']
        
        # 找到height和width在names中的索引位置
        height_idx = image_dim_names.index('height') if 'height' in image_dim_names else None
        width_idx = image_dim_names.index('width') if 'width' in image_dim_names else None
        
        if height_idx is None or width_idx is None:
            raise ValueError("Could not find 'height' or 'width' in image dimension names")
            
        # 根据索引从shape中获取实际的高度和宽度值
        image_shape = ds_meta.features[image_key]['shape']
        height = image_shape[height_idx]
        width = image_shape[width_idx]

        # image_transforms = ImageTransforms.create_piohfive_sequential_transform(
        #     (height, width)
        # ) if cfg.dataset.image_transforms.enable else None
        image_transforms = ImageTransforms.create_jax_pi0_main_camera_transform(img_size=(height, width)) if cfg.dataset.image_transforms.enable else None
        wrist_transforms = ImageTransforms.create_jax_pi0_wrist_camera_transform(img_size=(height, width)) if cfg.dataset.wrist_transforms.enable else None

        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        # breakpoint()

        # get total_episodes
        with open(f"{cfg.dataset.root}/meta/info.json", "r", encoding="utf-8") as f:
            data_cfg = json.load(f)
        print(f"total_episodes = {data_cfg['total_episodes']} !!!!")       
        total_episodes = data_cfg["total_episodes"] ###############################

        # 按比例划分 95% 训练，5% 测试
        train_rat = 0.999
        train_episodes = list(range(int(train_rat * total_episodes)))
        test_episodes = list(range(int(train_rat * total_episodes), total_episodes))

        dataset = cfn_lerobot_dataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=train_episodes,
            # episodes=cfg.dataset.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            wrist_transforms=wrist_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
            use_delta_action=cfg.policy.use_delta_action,
        )
        val_dataset = cfn_lerobot_dataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=test_episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            wrist_transforms=wrist_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
            use_delta_action=cfg.policy.use_delta_action,
        )
        # breakpoint()
    else:
        raise ValueError(f"Invalid dataset repo_id: {cfg.dataset.repo_id}")

    return dataset, val_dataset

def yang_eval(cfg, model, task, task2, replace_action, is_train_data):

    # task = "empty_cup_place"
    # task2 = "block_handover"
    # replace_action = 1
    # is_train_data = 1

    ckpt_task = "empty_cup_place"
    
    # weight_path = f'/gemini/platform/public/embodiedAI/users/ysy/data/train_cfn/trans-single_task-0812/{ckpt_task}-0812/model_epoch1.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.validate()

    ##########################################################################################################################################
    if is_train_data == 1:
    # 训练集
        cfg.dataset.root = f"/gemini/platform/public/embodiedAI/huggingface_cache/rhodes_lerobot/RoboTwin/eval_cfn/single_task/{task}_50epis"
    else:
    # 测试集
        cfg.dataset.root = f"/gemini/platform/public/embodiedAI/huggingface_cache/rhodes_lerobot/RoboTwin/eval_cfn/single_task2/{task}_25epis"
    dataset, _ = make_dataset(cfg)
    cfg.dataset.root = f"/gemini/platform/public/embodiedAI/huggingface_cache/rhodes_lerobot/RoboTwin/eval_cfn/single_task/{task2}_50epis"
    dataset2, _ = make_dataset(cfg)

    # if is_train_data == 1:
    # # 训练集
    #     cfg.dataset.root = "/gemini/platform/public/embodiedAI/huggingface_cache/rhodes_lerobot/RoboTwin/all_tasks_50ep"
    # else:
    # # 测试集
    #     cfg.dataset.root = f"/gemini/platform/public/embodiedAI/huggingface_cache/rhodes_lerobot/RoboTwin/eval_cfn/single_task2/{task}_25epis"
    # dataset, _ = make_dataset(cfg)
    # cfg.dataset.root = f"/gemini/platform/public/embodiedAI/huggingface_cache/rhodes_lerobot/RoboTwin/eval_cfn/single_task/{task2}_50epis"
    # dataset2, _ = make_dataset(cfg)
    ##########################################################################################################################################

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0, # cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dataloader2 = DataLoader(
        dataset2,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0, # cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    cfn_action_steps = 30

    # import ipdb; ipdb.set_trace()
    # batch['smol_inputs']['pixel_values'] = torch.zeros_like(batch['smol_inputs']['pixel_values'])

    # state = batch['observation.state'].to(device)
    # action = batch['action'][:, :cfn_action_steps, :].to(device)
    # task = batch['task']

    # import ipdb; ipdb.set_trace()

    # norm_task_dict = {'0': [], '1': [], '2': [], '3': [], '4': []}
    # norm_mean = [0] * 5
    with torch.no_grad():
        for i in range(5):
            # 随机取一个 batch
            batch = next(iter(dataloader))
            if task == task2:
                batch2 = next(iter(dataloader))
                # import ipdb; ipdb.set_trace()
            else:
                batch2 = next(iter(dataloader2))
            
            if replace_action == 1:
                batch['action'] = batch2['action']

            # batch['observation.state'] = torch.zeros_like(batch['observation.state'])
            # batch['action'] = torch.zeros_like(batch['action'])

            output = model(batch)  # (B, cfn_output_dim)
            # breakpoint()
            # 可以计算范数、均值等作为衡量
            output_norm = output.norm(dim=1)
            # import ipdb; ipdb.set_trace()

            # print(output_norm)
            torch.set_printoptions(precision=6)
            print(torch.mean(output_norm))

def yang_eval_nosie(cfg, model, data_task):

    # task = "empty_cup_place"
    # task2 = "block_handover"
    # replace_action = 1
    # is_train_data = 1

    data_task = data_task
    
    # weight_path = f'/gemini/platform/public/embodiedAI/users/ysy/data/train_cfn/trans-single_task-0812/{ckpt_task}-0812/model_epoch1.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.validate()

    cfg.dataset.root = f"/gemini/platform/public/embodiedAI/huggingface_cache/rhodes_lerobot/RoboTwin/eval_cfn/single_task/{data_task}_50epis"
    # cfg.dataset.root = "/gemini/platform/public/embodiedAI/huggingface_cache/rhodes_lerobot/RoboTwin/all_tasks_50ep"
    dataset, _ = make_dataset(cfg)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0, # cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    # import ipdb; ipdb.set_trace()
    # batch['smol_inputs']['pixel_values'] = torch.zeros_like(batch['smol_inputs']['pixel_values'])

    # state = batch['observation.state'].to(device)
    # action = batch['action'][:, :cfn_action_steps, :].to(device)
    # task = batch['task']
    

    # import ipdb; ipdb.set_trace()

    # norm_task_dict = {'0': [], '1': [], '2': [], '3': [], '4': []}
    # norm_mean = [0] * 5

    def sample_noise(shape, device, dtype):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=dtype,
            device=device,
        )
        return noise

    
    with torch.no_grad():
        # 随机取一个 batch
        batch = next(iter(dataloader)) 
        batch['action'] = batch['action'].to(device)
        noise_rate = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for i in range(len(noise_rate)):
            # batch['observation.state'] = torch.zeros_like(batch['observation.state'])
            # batch['action'] = torch.zeros_like(batch['action'])
            noise = sample_noise(batch['action'].shape, device, batch['action'].dtype)
            batch['action'] = batch['action'] * (1 - noise_rate[i]) + noise * noise_rate[i]

            output = model(batch)  # (B, cfn_output_dim)
            # breakpoint()
            # 可以计算范数、均值等作为衡量
            output_norm = output.norm(dim=1)
            # import ipdb; ipdb.set_trace()

            # print(output_norm)
            torch.set_printoptions(precision=6)
            print(torch.mean(output_norm))
            # print(output_norm)
            # print(output)
            # import ipdb; ipdb.set_trace()



@parser.wrap()
def test(cfg: TrainPipelineConfig):

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = CFNWrapper_pi_prior(
        cfn_output_dim=getattr(cfg.policy, "cfn_output_dim", 20),
        # pretrained_checkpoint_path="/gemini/platform/public/embodiedAI/users/ysy/data/dataset/rt_pi0_ckpt/25-07-21_12-18-18_pi0_gpu2_ck50_lr3e-5_bs12_s120K_seed42/checkpoints/060000/pretrained_model",
        pretrained_checkpoint_path="/gemini/platform/public/embodiedAI/users/ysy/data/dataset/rt_pi0_ckpt/robotwin_new_transforms_all_tasks_50ep/25-08-06_00-31-57_pi0_gpu4_ck50_lr3e-5_bs12_s60K_seed42/checkpoints/030000/pretrained_model",
    ).to(device)

    ckpt_task = "block_handover"
    
    weight_path = f"/gemini/platform/public/embodiedAI/users/ysy/data/train_cfn/cfn_pi-single_task-newckpt-prior-notrans-0828/{ckpt_task}-0828/model_epoch6.pt"
    # weight_path = f"/gemini/platform/public/embodiedAI/users/ysy/data/train_cfn/cfn_pi-single_task-0815/{ckpt_task}-0815/model_epoch1.pt"
    # 加载训练好的权重
    print(f"🔍 加载模型权重: {weight_path}")
    model.cfn.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # def yang_eval(cfg, model, task, task2, replace_action, is_train_data):
    yang_eval(cfg, model, "block_handover", "block_hammer_beat", 0, 1)

    # yang_eval_nosie(cfg, model, ckpt_task)

    import ipdb; ipdb.set_trace()
    print()



    # block_hammer_beat  block_handover     blocks_stack_easy     blocks_stack_hard   \
    # bottle_adjust      container_place    diverse_bottles_pick  dual_bottles_pick_easy  dual_bottles_pick_hard \
    # dual_shoes_place   empty_cup_place    mug_hanging_easy      mug_hanging_hard \
    # pick_apple_messy   put_apple_cabinet  shoe_place            tool_adjust
        


if __name__ == "__main__":
    test()
