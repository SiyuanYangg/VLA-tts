import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from transformers import set_seed
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.datasets.factory import make_dataset
from cfn_net import CFNWrapper


@parser.wrap()
def test(cfg: TrainPipelineConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    cfg.validate()

    # 加载数据
    dataset = make_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0, # cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    cfn_action_steps = 30

    model = CFNWrapper(
        state_dim=dataset[0]['observation.state'].shape[0],
        action_dim=cfn_action_steps * dataset[0]['action'].shape[1],
        language_model_name=getattr(cfg.policy, "language_model_name", "bert-base-uncased"),
        embed_dim=getattr(cfg.policy, "embed_dim", 128),
        cfn_output_dim=getattr(cfg.policy, "cfn_output_dim", 20),
    ).to(device)

    # breakpoint()

    # 加载训练好的权重
    weight_path = '/gemini/space/users/ysy/data/train_cfn/test-0722/model_epoch3.pt'
    weight_path = '/gemini/space/users/ysy/data/train_cfn/mlp-0723/model_epoch4.pt'
    print(f"🔍 加载模型权重: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 随机取一个 batch
    batch = next(iter(dataloader))
    batch = next(iter(dataloader))
    state = batch['observation.state'].to(device)
    action = batch['action'][:, :cfn_action_steps, :].to(device)
    task = batch['task']

    batch_size = action.shape[0]
    action_flat = action.reshape(batch_size, -1)

    # 不同噪声水平

    # import ipdb; ipdb.set_trace()

    
    norm_task_dict = {'0': [], '1': [], '2': [], '3': [], '4': []}
    norm_mean = [0] * 5
    with torch.no_grad():
        output = model(state, action_flat, task)  # (B, cfn_output_dim)
        # breakpoint()
        # 可以计算范数、均值等作为衡量
        output_norm = output.norm(dim=1)

        # import ipdb; ipdb.set_trace()

        for i in range(len(output_norm)):
            norm_task_dict[str(batch['task_index'][i].item())].append(output_norm[i].item())

        for i in range(5):
            norm_mean[i] = np.array(norm_task_dict[str(i)]).mean()
            count = 20 / (norm_mean[i] ** 2)
            print(f"task_index = {i}, norm_mean = {norm_mean[i]}, count = {count}")
        


if __name__ == "__main__":
    test()
