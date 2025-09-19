import torch
from cfn.pi0_cfn.cfn_net_pi_prior_big import CFNWrapper_pi_prior_big
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
    traj_num = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CFNWrapper_pi_prior_big(
        cfn_output_dim=20,
        pretrained_checkpoint_path="/gemini/platform/public/embodiedAI/users/ysy/data/dataset/rt_pi0_ckpt/robotwin_new_transforms_all_tasks_50ep/25-08-06_00-31-57_pi0_gpu4_ck50_lr3e-5_bs12_s60K_seed42/checkpoints/030000/pretrained_model",
    ).to(device)

    ckpt_task = task
    
    # weight_path = f"/gemini/platform/public/embodiedAI/users/ysy/data/train_cfn/{tag}/{ckpt_task}-0904/model_epoch8.pt"
    # weight_path = f"/gemini/platform/public/embodiedAI/users/ysy/data/train_cfn/{tag}/{ckpt_task}-0908/model_epoch16.pt"
    weight_path = f"/gemini/platform/public/embodiedAI/users/ysy/data/train_cfn/{tag}/{ckpt_task}/model_epoch16.pt"

    print(f"🔍 加载模型权重: {weight_path}")
    model.cfn.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # load index14 feature
    eval_task = ckpt_task
    features_dict = torch.load("/gemini/platform/public/embodiedAI/users/ysy/data/dataset/feature_rt/block_handover/feature.pt", map_location="cpu")
    features = []
    step_list = [9]
    for i in range(len(step_list)):
        features.append(features_dict[f"denoise_step{step_list[i]}"])
    features = torch.cat(features, dim=0)
    features = features.to(next(model.cfn.parameters()).dtype).to(next(model.cfn.parameters()).device)
    cfn_output = model.cfn(features[0:50])
    norm = cfn_output.norm(dim=1)

    def compare(model, features, index1, index2):
        feature1 = features[[index1]]
        feature2 = features[[index2]]
        feature_mid = (feature1 + feature2) / 2
        cfn_output1 = model.cfn(feature1)
        cfn_output2 = model.cfn(feature2)
        cfn_output_mid = model.cfn(feature_mid)
        print(f"index1 norm: {cfn_output1.norm(dim=1).item()}, index2 norm: {cfn_output2.norm(dim=1).item()}")
        print(f"mid norm: {cfn_output_mid.norm(dim=1).tolist()}")
        
    import ipdb;ipdb.set_trace()

    # data42 = torch.load(
    #     f"/gemini/platform/public/embodiedAI/users/ysy/data/robotwin/dataset/rt-record-traj/noised_action_50noise/noise_seed42/{eval_task}/traj_{traj_num}.pt",
    #     weights_only=False,
    # )
    # import ipdb;ipdb.set_trace()

    # get seed42 的50个noise
    # batch_size, noise_num, denoise_step, time_step, action_dim = data42["noised_action"].shape
    # device = data42["noised_action"].device
    # dtype = data42["noised_action"].dtype
    # actions_shape = (noise_num, model.policy.model.config.n_action_steps, model.policy.model.config.max_action_dim)

    seed = 42
    torch.manual_seed(seed)
    print(f"noise seed is {seed} !!!")
    # noise = self.policy.model.sample_noise(actions_shape, device, dtype)
    # noise = torch.normal(
    #     mean=0.0,
    #     std=1.0,
    #     size=actions_shape,
    #     dtype=dtype,
    # ).to(device)
    # print(f"noise is\n{noise}")

    # for i in range(batch_size):
    #     norm42 = {}

    #     # ======== forward seed42 ========
    #     batch42 = {
    #         # "action": data42["noised_action"][i, ...].reshape(noise_num*denoise_step, time_step, action_dim),
    #         "observation.state": data42["state"][[i], :].repeat(noise_num, 1),
    #         "observation.images.cam_high": data42["img_high"][[i], ...].repeat(noise_num, 1, 1, 1),
    #         "observation.images.cam_left_wrist": data42["img_left"][[i], ...].repeat(noise_num, 1, 1, 1),
    #         "observation.images.cam_right_wrist": data42["img_right"][[i], ...].repeat(noise_num, 1, 1, 1),
    #         "task": data42['lang'] * noise_num,
    #     }
    #     actions, features = model.policy.select_action_and_get_denoising_feature(batch42, noise.clone())
    #     features = features.to(next(model.cfn.parameters()).dtype)
    #     # import ipdb;ipdb.set_trace()
    #     features = features.reshape(noise_num*(denoise_step-1), 1024)
    #     cfn_output42 = model.cfn(features)
    #     norm42["norm"] = cfn_output42.norm(dim=1).reshape(noise_num, denoise_step-1)
    #     # import ipdb; ipdb.set_trace()
    #     norm42["is_suc"] = data42["is_suc"]
    #     norm42["now_seed"] = data42["now_seed"]

    #     # infer_step = data2345["noised_action"].shape[1]
    #     # batch2345 = {
    #     #     "action": data2345["noised_action"][i, ...],
    #     #     "observation.state": data2345["state"][[i], :].repeat(infer_step, 1),
    #     #     "observation.images.cam_high": data2345["img_high"][[i], ...].repeat(infer_step, 1, 1, 1),
    #     #     "observation.images.cam_left_wrist": data2345["img_left"][[i], ...].repeat(infer_step, 1, 1, 1),
    #     #     "observation.images.cam_right_wrist": data2345["img_right"][[i], ...].repeat(infer_step, 1, 1, 1),
    #     # }
    #     # cfn_output2345 = model.cfn(batch2345)
    #     # norm2345["norm"] = cfn_output2345.norm(dim=1).tolist()
    #     # norm2345["is_suc"] = data2345["is_suc"]
    #     # norm2345["now_seed"] = data2345["now_seed"]

    #     # # ======== forward seed2345 ========
    #     # batch2345 = {
    #     #     "action": data2345["action"],
    #     #     "observation.state": data2345["state"],
    #     #     "observation.images.cam_high": data2345["img_high"],
    #     #     "observation.images.cam_left_wrist": data2345["img_left"],
    #     #     "observation.images.cam_right_wrist": data2345["img_right"],
    #     # }
    #     # cfn_output2345 = model.cfn(batch2345)
    #     # norm2345["norm"] = cfn_output2345.norm(dim=1).tolist()
    #     # norm2345["is_suc"] = data2345["is_suc"]
    #     # norm2345["now_seed"] = data2345["now_seed"]

    #     # ======== 保存 json ========
    #     # os.makedirs(f"{tag}/noised_action_50noise/{eval_task}/seed42", exist_ok=True)
    #     # with open(f"{tag}/noised_action_50noise/{eval_task}/seed42/norm_infer_{i}.json", "w", encoding="utf-8") as f:
    #     #     json.dump(norm42, f, ensure_ascii=False, indent=4)
    #     # print(f"save at {tag}/noised_action_50noise/{eval_task}/seed42/norm_infer_{i}.json")

    #     # os.makedirs(f"{tag}/noised_action_50noise/{eval_task}/seed2345", exist_ok=True)
    #     # with open(f"{tag}/noised_action_50noise/{eval_task}/seed2345/norm_infer_{i}.json", "w", encoding="utf-8") as f:
    #     #     json.dump(norm2345, f, ensure_ascii=False, indent=4)
    #     # print(f"save at {tag}/noised_action_50noise/{eval_task}/seed2345/norm_infer_{i}.json")


    #     idx = torch.argmin(norm42["norm"][:, -1])
    #     # ======== 绘制单条轨迹对比 ========
    #     plt.figure(figsize=(8, 5))

    #     for j in range(1, noise_num):
    #         plt.plot(norm42["norm"][j].detach().cpu(), color="red")

    #     plt.plot(norm42["norm"][0].detach().cpu(), color="green")
    #     # plt.plot(norm42["norm"][idx].detach().cpu(), color="blue")

    #     # if norm2345["is_suc"]:
    #     #     plt.plot(norm2345["norm"], label="seed2345 (success)", color="blue")
    #     # else:
    #     #     plt.plot(norm2345["norm"], label="seed2345 (fail)", color="orange")

    #     plt.title(f"Norm inference {eval_task} #{i}")
    #     plt.xlabel("Denoise Step")
    #     plt.ylabel("Norm Value")
    #     plt.legend()
    #     plt.grid(True, linestyle="--", alpha=0.6)

    #     os.makedirs(f"{tag}/noised_action_50noise/{eval_task}/seed42/traj{traj_num}_is_suc{norm42['is_suc']}/plots", exist_ok=True)
    #     plt.savefig(f"{tag}/noised_action_50noise/{eval_task}/seed42/traj{traj_num}_is_suc{norm42['is_suc']}/plots/norm_infer_{i}.png", dpi=150)
    #     plt.close()

    # # ======== 绘制汇总图 ========
    # plot_summary(eval_task, tag)


if __name__ == "__main__":
    # for task in ["block_handover", "blocks_stack_easy", "container_place", "diverse_bottles_pick", "dual_bottles_pick_easy"]:
    for task in ["block_handover"]:
        # test(task, "cfn_pi-single_task-newckpt-prior-big-0905-good")
        # test(task, "cfn_pi-single_task-newckpt-prior-big-0908")
        # test(task, "cfn_pi-single_task-newckpt-prior-big-feature-0911")
        # test(task, "cfn_pi-single_task-newckpt-prior-big-feature2-step9-0911")
        # test(task, "cfn_pi-single_task-newckpt-prior-big-feature-step9-goodnoise-0911")
        # test(task, "cfn_pi-single_task-newckpt-prior-big-feature-step9-index14noise-0911")
        test(task, "aaa-0913/cfn_pi-single_task-newckpt-prior-big-feature-step9")
        
