import json
import os
import matplotlib.pyplot as plt

# test_index = 2
# tag = "cnncfn-logn-denoise-debug0904"
tag = "cnncfn-logn-denoise-debug0904-keystep10"
key_frame = 10
for test_index in range(1, 11):
    flie_dir = f"/gemini/platform/public/embodiedAI/users/ysy/data/robotwin/result_pi0/{tag}/block_handover/key_frame_infer_step{key_frame}/test_index{test_index}_norms.json"
    # 从 JSON 文件读取数据
    with open(flie_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    # import ipdb; ipdb.set_trace()
    # print("读取到的数据：", data)
    # print("姓名：", data["name"])

    total_infer_num = len(data['all_denoise_norms'])
    noise_num = len(data['all_denoise_norms'][0])
    eval_task = "block_handover"

    for i in range(total_infer_num):
        plt.figure(figsize=(8, 5))
        for j in range(noise_num):
            plt.plot(data['all_denoise_norms'][i][j], color="green")
        plt.plot(data['all_denoise_norms'][i][data['selected_index'][i]], color="red")
        plt.plot(data['all_denoise_norms'][i][0], color="blue")

        plt.title(f"Norm inference {eval_task} #{i}")
        plt.xlabel("Denoise Step")
        plt.ylabel("Norm Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        os.makedirs(f"{tag}/{eval_task}/seed42/test_index{test_index}_is_suc{data['is_suc']}", exist_ok=True)
        plt.savefig(f"{tag}/{eval_task}/seed42/test_index{test_index}_is_suc{data['is_suc']}/norm_infer_{i}.png", dpi=150)
        plt.close()
