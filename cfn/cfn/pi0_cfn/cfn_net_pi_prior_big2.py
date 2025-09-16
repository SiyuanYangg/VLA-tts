import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForVision2Seq
from torchvision.transforms import ToPILImage
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor
from transformers.models.idefics3.image_processing_idefics3 import Idefics3ImageProcessor

from cfn.action_emb import PerceiverConfig, SequentialActionEmb
import torch.nn.functional as F
from enum import Enum
from transformers.image_utils import ChannelDimension
import types

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.datasets.transforms import AbsoluteActionTransform
from torchvision.transforms import v2
import copy

class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(drop)
        self.norm = nn.LayerNorm(dim)

        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x


class CoinFlippingNetwork(nn.Module):
    """
    轻量增配版：
      - 默认 hidden_dim=256, depth=3, drop=0.1
      - 明显强于原始两层 MLP，但远小于千万级
    """
    def __init__(self, input_dim, output_dim=64, hidden_dim=256, depth=3, drop=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_act = nn.GELU()
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            MLPBlock(hidden_dim, hidden_dim * 4, drop=drop)
            for _ in range(depth)
        ])

        self.output = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output.weight);     nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_act(x)
        x = self.input_norm(x)
        for blk in self.blocks:
            x = blk(x)
        return self.output(x)



# ========== CFN 主网络 ========== #
# class CoinFlippingNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim=64):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, 32)
#         self.fc2 = nn.Linear(32, 20)
#         self.output = nn.Linear(20, output_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.output(x)


class cfn_net(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.cfn = CoinFlippingNetwork(input_dim=input_dim, output_dim=output_dim)

        # Deep copy for prior model
        self.prior_cfn = copy.deepcopy(self.cfn)
        for param in self.prior_cfn.parameters():
            param.requires_grad = False
        
        prior_mean = torch.nn.Parameter(torch.zeros(1, output_dim), requires_grad=False)
        prior_var = torch.nn.Parameter(torch.full((1, output_dim), 0.01), requires_grad=False)
        self.register_buffer("prior_mean", prior_mean)
        self.register_buffer("prior_var", prior_var)

        # Buffer to store prior outputs
        self.prior_outputs = None
        self.iter_count = 0
        self.output_dim = output_dim

    def update_prior(self):
        """Incrementally update the prior mean and variance."""
        for i in range(self.prior_outputs.shape[0]):
            self.iter_count += 1
            
            # Compute delta between the new output and current mean
            delta = self.prior_outputs[i] - self.prior_mean
            
            # Update mean: mean = mean + delta / iter_count
            self.prior_mean.data += delta / self.iter_count
            
            # Update variance: variance = variance + delta^2 / iter_count
            delta_sq = delta ** 2
            self.prior_var.data += (delta_sq - self.prior_var) / self.iter_count


    def forward(self, features):
        # Record prior output (called during forward pass to update the buffer)
        with torch.no_grad():
            self.prior_outputs = self.prior_cfn(features)
            prior = (self.prior_outputs - self.prior_mean) / torch.sqrt(self.prior_var + 1e-6)
        coin_flipping = self.cfn(features) + torch.ones_like(prior)
        self.prior = prior
        return coin_flipping



# ========== CFN 包装类（支持图文编码） ========== #
class CFNWrapper_pi_prior_big(nn.Module):
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

        # import ipdb; ipdb.set_trace()

        # self.vision_model.eval()
        # for p in self.vision_model.parameters():
        #     p.requires_grad = False

        # hidden_size = self.vision_model.config.text_config.hidden_size
        # self.language_proj = nn.Linear(hidden_size, embed_dim)

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

        with torch.no_grad():
            actions, features = self.policy.get_feature(batch)
            features = features.to(next(self.parameters()).dtype)
        # import ipdb; ipdb.set_trace()
        return self.cfn(features)
    
    def compute_loss(self, batch):
        preds = self.forward(batch)
        self.cfn.update_prior()
        targets = batch["CoinFlip_target"].float().to(next(self.parameters()).device)
        loss = F.mse_loss(preds, targets)
        return loss, preds

    def compute_loss_feature(self, feature_batch):
        for key in feature_batch:
            if isinstance(feature_batch[key], torch.Tensor):
                feature_batch[key] = feature_batch[key].to(next(self.parameters()).device)
        feature_batch["feature"] = feature_batch["feature"].to(next(self.parameters()).dtype)
        preds = self.cfn(feature_batch["feature"])
        self.cfn.update_prior()
        targets = feature_batch["CoinFlip_target"].float().to(next(self.parameters()).device)
        # import ipdb;ipdb.set_trace()
        loss = F.mse_loss(preds, targets)
        return loss, preds
