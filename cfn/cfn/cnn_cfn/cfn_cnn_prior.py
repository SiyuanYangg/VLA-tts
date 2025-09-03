import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from enum import Enum

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import copy


# ========== CFN 主网络 ========== #
class CoinFlippingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 20)
        self.output = nn.Linear(20, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

class FeatureEncoder(nn.Module):
    """编码器：动作 MLP + 状态 MLP + 图像 CNN"""
    def __init__(self, cnn_out_dim=256, mlp_hidden_dim=128):
        super().__init__()
        # 动作编码 [b, 50, 14]
        self.action_mlp = nn.Sequential(
            nn.Linear(50 * 14, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )
        # 状态编码 [b, 14]
        self.state_mlp = nn.Sequential(
            nn.Linear(14, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )
        # 简单 CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.cnn_fc = nn.Linear(128, cnn_out_dim)

    def forward(self, batch):
        b = batch["action"].shape[0]

        # 动作编码
        action_feat = self.action_mlp(batch["action"].reshape(b, -1))
        # 状态编码
        state_feat = self.state_mlp(batch["observation.state"])

        # 三路相机图像编码
        img_feats = []
        for cam in ["observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist"]:
            x = self.cnn(batch[cam])        # [b, 128, 1, 1]
            x = x.view(x.size(0), -1)       # [b, 128]
            x = self.cnn_fc(x)              # [b, cnn_out_dim]
            img_feats.append(x)
        img_feat = torch.cat(img_feats, dim=-1)  # [b, 3*cnn_out_dim]

        # 拼接所有模态
        features = torch.cat([action_feat, state_feat, img_feat], dim=-1)
        return features


class cfn_net(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        # feature encoder
        self.encoder = FeatureEncoder(cnn_out_dim=256, mlp_hidden_dim=128)

        # CFN 主网络
        self.cfn = CoinFlippingNetwork(input_dim=input_dim, output_dim=output_dim)

        # Deep copy for prior model
        self.prior_cfn = copy.deepcopy(self.cfn)
        for param in self.prior_cfn.parameters():
            param.requires_grad = False
        self.prior_encoder = copy.deepcopy(self.encoder)
        for param in self.prior_encoder.parameters():
            param.requires_grad = False
        
        prior_mean = torch.nn.Parameter(torch.zeros(1, output_dim), requires_grad=False)
        prior_var = torch.nn.Parameter(torch.full((1, output_dim), 0.0001), requires_grad=False)
        self.register_buffer("prior_mean", prior_mean)
        self.register_buffer("prior_var", prior_var)

        self.prior_outputs = None
        self.iter_count = 0
        self.output_dim = output_dim

    def update_prior(self):
        if self.prior_outputs is None:
            return
        for i in range(self.prior_outputs.shape[0]):
            self.iter_count += 1
            delta = self.prior_outputs[i] - self.prior_mean
            self.prior_mean.data += delta / self.iter_count
            delta_sq = delta ** 2
            self.prior_var.data += (delta_sq - self.prior_var) / self.iter_count

    def forward(self, batch):
        # 把 batch 编码成 features
        features = self.encoder(batch)

        with torch.no_grad():
            features_prior = self.prior_encoder(batch)
            self.prior_outputs = self.prior_cfn(features_prior)
            prior = (self.prior_outputs - self.prior_mean) / torch.sqrt(self.prior_var + 1e-6)

        coin_flipping = self.cfn(features) + prior / torch.sqrt(torch.tensor(self.output_dim, device=features.device))
        self.prior = prior
        return coin_flipping


# ========== CFN 包装类（支持图文编码） ========== #
class CFNWrapper_cnn_prior(nn.Module):
    def __init__(self,
                 cfn_output_dim=64,
                 cfn_action_steps=None,
                 pretrained_checkpoint_path=None
                 ):
        super().__init__()

        print(f"loading pretrained checkpoint from {pretrained_checkpoint_path}...")
        self.policy = PI0Policy.from_pretrained(pretrained_checkpoint_path, local_files_only=True).eval()
        print("loading model success!")

        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False

        # 多模态融合预测模块
        self.cfn = cfn_net(
            input_dim=1024,
            output_dim=cfn_output_dim
        ).to(next(self.policy.parameters()).device)


    def forward(self, batch):
        # import ipdb; ipdb.set_trace()
        # breakpoint()

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(next(self.parameters()).device)
        
        return self.cfn(batch)

    def compute_loss(self, batch):
        preds = self.forward(batch)
        self.cfn.update_prior()
        targets = batch["CoinFlip_target"].float().to(next(self.parameters()).device)
        loss = F.mse_loss(preds, targets)
        return loss, preds


