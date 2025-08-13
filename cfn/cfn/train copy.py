import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import set_seed
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import time

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import cycle
from lerobot.common.utils.logging_utils import AverageMeter
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from cfn_net import CFNWrapper

class CoinFlipMaker:
    def __init__(self, output_dimensions=20, only_zero_flips=False):
        self.output_dimensions = output_dimensions
        self.only_zero_flips = only_zero_flips

    def __call__(self, batch_size):
        if self.only_zero_flips:
            return np.zeros((batch_size, self.output_dimensions), dtype=np.float32)
        return 2 * np.random.binomial(1, 0.5, size=(batch_size, self.output_dimensions)) - 1


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    overall_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    cfg.validate()

    t0 = time.time()
    dataset = make_dataset(cfg)
    t1 = time.time()
    print(f"📦 数据集加载时间: {t1 - t0:.2f}s")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    # breakpoint()
    dataloader = cycle(dataloader)

    # accum_data_time = 0
    # for i in range(500):
    #     data_start = time.time()
    #     batch = next(dataloader)
    #     data_end = time.time()
    #     accum_data_time += data_end - data_start
    #     if i % 5 == 0:
    #         print(f"data={accum_data_time:.2f}s")
    #         accum_data_time = 0
    # breakpoint()

    cfn_action_steps = 20
    assert len(dataset[0]['observation.state'].shape) == 1
    assert len(dataset[0]['action'].shape) == 2
    assert cfn_action_steps <= dataset[0]['action'].shape[0]

    t0 = time.time()
    model = CFNWrapper(
        state_dim=dataset[0]['observation.state'].shape[0],
        action_dim=cfn_action_steps * dataset[0]['action'].shape[1],
        language_model_name=getattr(cfg.policy, "language_model_name", "bert-base-uncased"),
        embed_dim=getattr(cfg.policy, "embed_dim", 128),
        cfn_output_dim=getattr(cfg.policy, "cfn_output_dim", 20),
    ).to(device)
    t1 = time.time()
    print(f"🧠 模型初始化时间: {t1 - t0:.2f}s")

    coin_flip_maker = CoinFlipMaker(output_dimensions=model.cfn.output.out_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    loss_meter = AverageMeter("loss", ":.4f")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    num_batches_per_epoch = len(dataset) // cfg.batch_size
    total_epochs = cfg.steps // num_batches_per_epoch + 1

    for epoch in range(total_epochs):
        print(f"\n📘 Epoch {epoch + 1} 开始")
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        # 计时累加器
        accum_batch_time = 0.0
        accum_data_time = 0.0
        accum_forward_time = 0.0
        accum_backward_time = 0.0
        batch_counter = 0

        for i in tqdm(range(num_batches_per_epoch), desc=f"Epoch {epoch+1}"):
            batch_start = time.time()

            data_start = time.time()
            batch = next(dataloader)
            data_end = time.time()

            state = batch['observation.state'].to(device)
            action = batch["action"].to(device)
            instructions = batch["task"]
            action = action[:, :cfn_action_steps, :]
            batch_size = action.shape[0]
            action = action.reshape(batch_size, -1)

            target_np = coin_flip_maker(batch_size=state.size(0))
            target = torch.from_numpy(target_np).float().to(device)

            forward_start = time.time()
            loss = model.compute_loss(state, action, instructions, target)
            forward_end = time.time()

            backward_start = time.time()
            loss.backward()
            backward_end = time.time()

            batch_end = time.time()

            # 累积时间
            accum_batch_time += batch_end - batch_start
            accum_data_time += data_end - data_start
            accum_forward_time += forward_end - forward_start
            accum_backward_time += backward_end - backward_start
            batch_counter += 1

            total_loss += loss.item()

            # 每 50 个 batch 打印一次平均耗时
            if batch_counter == 200:
                print(f"⏱️ 平均每 200 batch 用时: "
                      f"batch={accum_batch_time:.2f}s | "
                      f"data={accum_data_time:.2f}s | "
                      f"forward={accum_forward_time:.2f}s | "
                      f"backward={accum_backward_time:.2f}s")
                # 重置
                accum_batch_time = 0.0
                accum_data_time = 0.0
                accum_forward_time = 0.0
                accum_backward_time = 0.0
                batch_counter = 0

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = total_loss / num_batches_per_epoch
        loss_meter.update(avg_loss)
        print(f"✅ Epoch {epoch + 1} 平均 Loss: {avg_loss:.4f}")
        print(f"🕒 Epoch 总耗时: {time.time() - epoch_start:.2f}s")

        if (epoch + 1) % (cfg.save_freq // num_batches_per_epoch) == 0:
            torch.save(model.state_dict(), output_dir / f"model_epoch{epoch+1}.pt")

    print(f"✅ 所有训练完成，总耗时: {time.time() - overall_start:.2f}s")


if __name__ == "__main__":
    train()
