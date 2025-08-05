import argparse
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import torch
import random

def inspect_dataset(path: str):
    print(f"🔍 正在加载数据集: {path}")
    ds = LeRobotDataset(repo_id=None, root=Path(path))
    
    print(f"✅ 成功加载数据集: {ds.repo_id}")
    print(f"- 总 episodes: {ds.num_episodes}")
    print(f"- 总帧数: {ds.num_frames}")
    print(f"- fps: {ds.fps}")
    print(f"- features: {list(ds.features.keys())}")
    print(f"- 摄像头: {ds.meta.video_keys}")
    print(f"- tasks: {len(ds.meta.tasks)} 个任务标签")

    print("\n🎞️ 预览前 5 帧并保存图像：")
    save_dir = Path("preview")
    save_dir.mkdir(exist_ok=True)


    frame_indices = random.sample(range(len(ds)), k=min(10, len(ds)))
    for i in frame_indices:
        frame = ds[i]
        print(f"--- Frame {i} ---")
        print(f"任务: {frame['task']}")
        print(f"timestamp: {frame['timestamp']}")
        print(f"action shape: {frame['action'].shape}")
        print(f"observation.state shape: {frame['observation.state'].shape}")

        # breakpoint()
        # 保存图像帧（只保存一个摄像头）
        # for cam in ds.meta.video_keys[:1]:
        cam = 'observation.images.cam_high'
        if cam in frame:
            img = frame[cam]
            if isinstance(img, PIL.Image.Image):
                img = np.array(img)
            
            if isinstance(img, torch.Tensor):
                if img.shape[0] == 3:  # (C, H, W)
                    img = img.permute(1, 2, 0).cpu().numpy()  # → (H, W, C)
            elif isinstance(img, np.ndarray):
                if img.shape[0] == 3:  # (C, H, W)
                    img = img.transpose(1, 2, 0)  # → (H, W, C)
            elif isinstance(img, PIL.Image.Image):
                img = np.array(img)            
            plt.imshow(img)
            plt.title(f"{cam} - Frame {i}")
            plt.axis("off")
            out_path = save_dir / f"{cam.replace('.', '_')}_frame_{i}.png"
            plt.savefig(out_path)
            plt.close()
            print(f"✅ 已保存图像到: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="Path to your LeRobot dataset")
    args = parser.parse_args()
    inspect_dataset(args.dataset_path)
