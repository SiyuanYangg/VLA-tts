import torch
import numpy as np
from torch.utils.data import DataLoader

from transformers import set_seed
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.datasets.factory import make_dataset
from cfn_net import CFNWrapper


@parser.wrap()
def test(cfg: TrainPipelineConfig):
    # 设置设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    cfg.validate()

    # 加载数据
    dataset = make_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    cfn_action_steps = 30

    # 初始化模型
    model = CFNWrapper(
        state_dim=dataset[0]['observation.state'].shape[0],
        action_dim=cfn_action_steps * dataset[0]['action'].shape[1],
        language_model_name=getattr(cfg.policy, "language_model_name", "bert-base-uncased"),
        embed_dim=getattr(cfg.policy, "embed_dim", 128),
        cfn_output_dim=getattr(cfg.policy, "cfn_output_dim", 20),
    ).to(device)

    # 加载权重
    weight_path = '/gemini/space/users/ysy/data/train_cfn/mlp/model_epoch1.pt'
    print(f"🔍 加载模型权重: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 遍历多个 batch 收集推理结果
    n = 5  # 要处理的 batch 数量
    action_flats = []
    output_norms = []

    print(f"📦 正在逐 batch 推理并收集 {n} 个 batch 的输出 ...")
    for i, batch in enumerate(dataloader):
        if i >= n:
            break
        state = batch['observation.state'].to(device)
        action = batch['action'][:, :cfn_action_steps, :].to(device)
        task = batch['task']

        with torch.no_grad():
            action_flat = action.reshape(action.shape[0], -1)
            output = model(state, action_flat, task)
            norm = output.norm(dim=1).cpu().numpy()  # (B,)
            action_flat_cpu = action_flat.cpu().numpy()  # (B, D)

        action_flats.append(action_flat_cpu)
        output_norms.append(norm)

    # 合并所有 batch 的结果
    action_flats = np.concatenate(action_flats, axis=0)  # (N, D)
    output_norms = np.concatenate(output_norms, axis=0)  # (N,)

    print("✅ 成功获取所有模型输出")

    # PCA 降维
    pca = PCA(n_components=2)
    action_2d = pca.fit_transform(action_flats)  # (N, 2)
    x, y = action_2d[:, 0], action_2d[:, 1]
    z = output_norms

    # 创建热力图网格
    grid_size = 20
    x_edges = np.linspace(x.min(), x.max(), grid_size)
    y_edges = np.linspace(y.min(), y.max(), grid_size)

    heatmap = np.full((grid_size - 1, grid_size - 1), np.nan)
    count = np.zeros_like(heatmap)

    print("📊 构建热力图数据 ...")
    for xi, yi, zi in zip(x, y, z):
        x_idx = np.searchsorted(x_edges, xi) - 1
        y_idx = np.searchsorted(y_edges, yi) - 1
        if 0 <= x_idx < grid_size - 1 and 0 <= y_idx < grid_size - 1:
            if np.isnan(heatmap[y_idx, x_idx]):
                heatmap[y_idx, x_idx] = zi
                count[y_idx, x_idx] = 1
            else:
                heatmap[y_idx, x_idx] += zi
                count[y_idx, x_idx] += 1

    # 区域平均
    heatmap = heatmap / np.where(count == 0, 1, count)

    # 设置 colormap，将 NaN 显示为透明
    cmap = cm.get_cmap('viridis').copy()
    cmap.set_bad(color=(1, 1, 1, 0))  # 透明

    print("🖼️ 绘制图像 ...")
    plt.figure(figsize=(8, 6))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    plt.imshow(heatmap, origin='lower', extent=extent, aspect='auto', cmap=cmap, alpha=0.9)

    # 空心白圈表示动作点
    plt.scatter(x, y, facecolors='none', edgecolors='white', s=30, linewidths=0.8)

    plt.title('PCA of Actions with Output Norm Heatmap')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.colorbar(label='Average Output Norm in Region')
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    save_path = './pca_output_heatmap2.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图像已保存至 {save_path}")


if __name__ == "__main__":
    test()
