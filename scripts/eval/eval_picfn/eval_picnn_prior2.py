import torch
from cfn.pi0_cfn.cfn_net_pi_prior2 import CFNWrapper_pi_prior
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def pad_to_max(arr_list, max_len):
    """把不同长度的轨迹 pad 到相同长度，缺失补 NaN"""
    padded = []
    for seq in arr_list:
        seq = np.array(seq, dtype=float)
        if len(seq) < max_len:
            seq = np.pad(seq, (0, max_len - len(seq)), constant_values=np.nan)
        padded.append(seq)
    return np.vstack(padded)


def plot_summary(eval_task, tag):
    """画 seed42 成功 和 seed2345 失败 的均值±方差曲线"""
    seed42_norms = []
    seed2345_norms = []

    for i in range(20):
        with open(f"{tag}/{eval_task}/seed42/norm_traj_{i}.json", "r", encoding="utf-8") as f:
            norm42 = json.load(f)
        with open(f"{tag}/{eval_task}/seed2345/norm_traj_{i}.json", "r", encoding="utf-8") as f:
            norm2345 = json.load(f)

        if norm42["is_suc"]:
            seed42_norms.append(norm42["norm"])
        if not norm2345["is_suc"]:
            seed2345_norms.append(norm2345["norm"])

    # 打印数量统计
    print(f"✅ seed42 成功轨迹数量: {len(seed42_norms)}")
    print(f"❌ seed2345 失败轨迹数量: {len(seed2345_norms)}")

    if not seed42_norms and not seed2345_norms:
        print("⚠️ 没有符合条件的轨迹，跳过绘制")
        return

    # 取最长长度并 padding
    max_len_42 = max((len(x) for x in seed42_norms), default=0)
    max_len_2345 = max((len(x) for x in seed2345_norms), default=0)

    seed42_array = pad_to_max(seed42_norms, max_len_42) if seed42_norms else np.array([])
    seed2345_array = pad_to_max(seed2345_norms, max_len_2345) if seed2345_norms else np.array([])

    # 计算均值和标准差（忽略 NaN）
    mean42 = np.nanmean(seed42_array, axis=0) if seed42_norms else []
    std42 = np.nanstd(seed42_array, axis=0) if seed42_norms else []

    mean2345 = np.nanmean(seed2345_array, axis=0) if seed2345_norms else []
    std2345 = np.nanstd(seed2345_array, axis=0) if seed2345_norms else []

    # 绘图
    plt.figure(figsize=(8, 5))

    if len(mean42) > 0:
        x42 = np.arange(len(mean42))
        plt.plot(x42, mean42, color="green", label="seed42 success (mean)")
        plt.fill_between(x42, mean42 - std42, mean42 + std42, color="green", alpha=0.3)

    if len(mean2345) > 0:
        x2345 = np.arange(len(mean2345))
        plt.plot(x2345, mean2345, color="red", label="seed2345 fail (mean)")
        plt.fill_between(x2345, mean2345 - std2345, mean2345 + std2345, color="red", alpha=0.3)

    plt.title(f"Summary of Norm Trajectories ({eval_task})")
    plt.xlabel("Step")
    plt.ylabel("Norm Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    os.makedirs(f"{tag}/{eval_task}/plots", exist_ok=True)
    plt.savefig(f"{tag}/{eval_task}/plots/summary_norm.png", dpi=150)
    plt.close()


def test(task, tag):
    tag = tag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CFNWrapper_pi_prior(
        cfn_output_dim=20,
        pretrained_checkpoint_path="/gemini/platform/public/embodiedAI/users/ysy/data/dataset/rt_pi0_ckpt/robotwin_new_transforms_all_tasks_50ep/25-08-06_00-31-57_pi0_gpu4_ck50_lr3e-5_bs12_s60K_seed42/checkpoints/030000/pretrained_model",
    ).to(device)

    ckpt_task = task
    weight_path = f"/gemini/platform/public/embodiedAI/users/ysy/data/train_cfn/{tag}/{ckpt_task}-0901/model_epoch8.pt"
    print(f"🔍 加载模型权重: {weight_path}")
    model.cfn.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    eval_task = ckpt_task
    for i in range(20):
        norm42 = {}
        norm2345 = {}

        data42 = torch.load(
            f"/gemini/platform/public/embodiedAI/users/ysy/data/robotwin/dataset/rt-record-traj/noise_seed42/{eval_task}/traj_{i}.pt",
            weights_only=False,
        )
        data2345 = torch.load(
            f"/gemini/platform/public/embodiedAI/users/ysy/data/robotwin/dataset/rt-record-traj/noise_seed2345/{eval_task}/traj_{i}.pt",
            weights_only=False,
        )

        # ======== forward seed42 ========
        batch42 = {
            "action": data42["action"],
            "observation.state": data42["state"],
            "observation.images.cam_high": data42["img_high"],
            "observation.images.cam_left_wrist": data42["img_left"],
            "observation.images.cam_right_wrist": data42["img_right"],
            "task": data42["lang"] * data42["action"].shape[0],
        }
        cfn_output42 = model(batch42)
        norm42["norm"] = cfn_output42.norm(dim=1).tolist()
        norm42["is_suc"] = data42["is_suc"]
        norm42["now_seed"] = data42["now_seed"]

        # ======== forward seed2345 ========
        batch2345 = {
            "action": data2345["action"],
            "observation.state": data2345["state"],
            "observation.images.cam_high": data2345["img_high"],
            "observation.images.cam_left_wrist": data2345["img_left"],
            "observation.images.cam_right_wrist": data2345["img_right"],
            "task": data2345["lang"] * data2345["action"].shape[0],
        }
        cfn_output2345 = model(batch2345)
        norm2345["norm"] = cfn_output2345.norm(dim=1).tolist()
        norm2345["is_suc"] = data2345["is_suc"]
        norm2345["now_seed"] = data2345["now_seed"]

        # ======== 保存 json ========
        os.makedirs(f"{tag}/{eval_task}/seed42", exist_ok=True)
        with open(f"{tag}/{eval_task}/seed42/norm_traj_{i}.json", "w", encoding="utf-8") as f:
            json.dump(norm42, f, ensure_ascii=False, indent=4)

        os.makedirs(f"{tag}/{eval_task}/seed2345", exist_ok=True)
        with open(f"{tag}/{eval_task}/seed2345/norm_traj_{i}.json", "w", encoding="utf-8") as f:
            json.dump(norm2345, f, ensure_ascii=False, indent=4)

        # ======== 绘制单条轨迹对比 ========
        plt.figure(figsize=(8, 5))

        if norm42["is_suc"]:
            plt.plot(norm42["norm"], label="seed42 (success)", color="green")
        else:
            plt.plot(norm42["norm"], label="seed42 (fail)", color="red")

        if norm2345["is_suc"]:
            plt.plot(norm2345["norm"], label="seed2345 (success)", color="blue")
        else:
            plt.plot(norm2345["norm"], label="seed2345 (fail)", color="orange")

        plt.title(f"Norm Trajectory {eval_task} #{i}")
        plt.xlabel("Step")
        plt.ylabel("Norm Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        os.makedirs(f"{tag}/{eval_task}/plots", exist_ok=True)
        plt.savefig(f"{tag}/{eval_task}/plots/norm_traj_{i}.png", dpi=150)
        plt.close()

    # ======== 绘制汇总图 ========
    plot_summary(eval_task, tag)


if __name__ == "__main__":
    for task in ["block_handover", "blocks_stack_easy", "container_place", "diverse_bottles_pick", "dual_bottles_pick_easy"]:
        test(task, "cfn_pi-single_task-newckpt-prior-0901")
