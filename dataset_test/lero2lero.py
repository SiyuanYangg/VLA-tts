

import random
from pathlib import Path
from datasets import load_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

# 设置原始和目标数据集路径
orig_root = Path("/gemini/space/huggingface_cache/rhodes_lerobot/RoboTwin/all_tasks_50ep")
new_root = Path("/gemini/space/users/ysy/data/dataset/lerobot_robotwin_dataset")

# 加载原始数据集（仅加载元信息）
orig = LeRobotDataset(repo_id=None, root=orig_root)

# 从所有 episode index 中随机抽取 10 个
all_episodes = list(range(orig.meta.total_episodes))
selected_eps = random.sample(all_episodes, k=5)
print("随机选择的 episodes:", selected_eps)

# 创建新数据集，fps 和 features 与原始一致
new_ds = LeRobotDataset.create(
    repo_id="test/7_22",
    fps=orig.fps,
    root=new_root,
    features=orig.meta.features,
    use_videos=True,
)

# 遍历每个被选中的 episode，逐帧复制
num = 1
for new_ep_idx, src_ep_idx in enumerate(selected_eps):
    parquet_path = orig_root / orig.meta.get_data_file_path(src_ep_idx)
    data = load_dataset("parquet", data_files=str(parquet_path), split="train")
    
    # 新 episode buffer 初始化
    # new_ds.episode_buffer = new_ds.create_episode_buffer(episode_index=new_ep_idx)
    
    # breakpoint()
    task=''
    for i in range(num):
        for row in data:
            new_ds.add_frame({
                "observation.state": np.array(row["observation.state"], dtype=np.float32),
                "action": np.array(row["action"], dtype=np.float32),
                # "timestamp": np.array(row["timestamp"]),
                "task": orig.meta.tasks[row["task_index"]],
                # 图像字段（可选）
                "observation.images.cam_high": row["observation.images.cam_high"],
                "observation.images.cam_left_wrist": row["observation.images.cam_left_wrist"],
                "observation.images.cam_right_wrist": row["observation.images.cam_right_wrist"],
            })
            task=orig.meta.tasks[row["task_index"]]
        new_ds.save_episode()  # 保存这个 episode

    print(f"save episode {src_ep_idx}, times: {num}, task: {task}")
    
    num += 1
    # print(f"保存 episode {new_ep_idx} 完成（来自原始 episode {src_ep_idx}）")

print("✅ 所有 10 个 episodes 已保存为新数据集")


