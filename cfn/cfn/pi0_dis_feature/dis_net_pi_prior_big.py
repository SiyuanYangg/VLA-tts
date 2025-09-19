import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ========== 基础 MLPBlock ========== #
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


# ========== 主干网络 ========== #
class DistanceNetwork(nn.Module):
    """
    轻量增配版：
      - 默认 hidden_dim=256, depth=3, drop=0.1
      - 输出 2*output_dim (mu, log_var)
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

        # 输出层：均值 mu + log_var
        self.output = nn.Linear(hidden_dim, output_dim * 2)

        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output.weight);     nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_act(x)
        x = self.input_norm(x)
        for blk in self.blocks:
            x = blk(x)
        return self.output(x)


# ========== dis_net with constant prior ========== #
class dis_net(nn.Module):
    def __init__(self, input_dim,
                 prior_mean_val=5.0, prior_logvar_val=3.0):
        super().__init__()
        self.dis = DistanceNetwork(input_dim=input_dim, output_dim=2)

        # 常数 prior
        self.register_buffer("prior_mean", torch.tensor(prior_mean_val))
        self.register_buffer("prior_logvar", torch.tensor(prior_logvar_val))

    def forward(self, features):
        # features = features * 10
        
        out = self.dis(features)
        mu, log_var = out[:, 0], out[:, 1]

        # 叠加 prior 偏移
        prior_mu = self.prior_mean
        prior_logvar = self.prior_logvar

        mu = mu + prior_mu
        log_var = log_var + prior_logvar

        return mu, log_var


# ========== 包装类 ========== #
class DISWrapper_pi_prior_big(nn.Module):
    def __init__(self,
                 pretrained_checkpoint_path=None,
                 prior_mean_val=5.0,
                 prior_logvar_val=1.0):
        super().__init__()

        print(f"loading pretrained checkpoint from {pretrained_checkpoint_path}...")
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        self.policy = PI0Policy.from_pretrained(pretrained_checkpoint_path, local_files_only=True).eval()
        print("loading model success!")

        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False

        # 使用常数 prior 的 dis_net
        self.dis = dis_net(
            input_dim=1024,
            prior_mean_val=prior_mean_val,
            prior_logvar_val=prior_logvar_val
        ).to(next(self.policy.parameters()).device)

    def forward(self, batch):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(next(self.parameters()).device)

        with torch.no_grad():
            actions, features = self.policy.get_feature(batch)
            features = features.to(next(self.parameters()).dtype)

        return self.dis(features)  # (mu, log_var)

    def nll_loss(self, mu, log_var, targets):
        # 高斯 NLL
        loss = 0.5 * (torch.exp(-log_var) * (targets - mu) ** 2 + log_var)
        return loss.mean()

    def compute_loss(self, batch):
        mu, log_var = self.forward(batch)
        targets = batch["distance"].float().to(next(self.parameters()).device)
        loss = self.nll_loss(mu, log_var, targets)
        return loss, mu, log_var

    def compute_loss_feature(self, feature_batch):
        for key in feature_batch:
            if isinstance(feature_batch[key], torch.Tensor):
                feature_batch[key] = feature_batch[key].to(next(self.parameters()).device)
        feature_batch["feature"] = feature_batch["feature"].to(next(self.parameters()).dtype)

        mu, log_var = self.dis(feature_batch["feature"])
        targets = feature_batch["distance"].float().to(next(self.parameters()).device)
        loss = self.nll_loss(mu, log_var, targets)
        # import ipdb;ipdb.set_trace()
        return loss, mu, log_var
