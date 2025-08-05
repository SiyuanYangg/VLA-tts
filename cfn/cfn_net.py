import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")

# ========== MLP 编码器 ========== #
class SimpleMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # breakpoint()
        return self.model(x)

# ========== CFN ========== #
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

# ========== CFN Wrapper ========== #
class CFNWrapper(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 language_model_name='bert-base-uncased',
                 embed_dim=128,
                 cfn_output_dim=64,
                 ):
        super().__init__()

        # Action/State 编码器
        self.state_encoder = SimpleMLPEncoder(state_dim, [64, 64], embed_dim)
        self.action_encoder = SimpleMLPEncoder(action_dim, [64, 64], embed_dim)

        # 文本编码器（冻结参数）
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.text_encoder = AutoModel.from_pretrained(language_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.language_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        # CFN
        total_feature_dim = embed_dim * 3
        self.cfn = CoinFlippingNetwork(input_dim=total_feature_dim,
                                       output_dim=cfn_output_dim)

    def encode_instruction(self, instruction_list):
        tokens = self.tokenizer(instruction_list, padding=True, truncation=True,
                                return_tensors="pt").to(next(self.parameters()).device)
        with torch.no_grad():
            outputs = self.text_encoder(**tokens).last_hidden_state
            cls_embedding = outputs[:, 0, :]  # [CLS]
        return self.language_proj(cls_embedding)  # shape: (batch_size, embed_dim)

    def forward(self, state, action, instruction_list):
        state_feat = self.state_encoder(state)              # (B, embed_dim)
        # breakpoint()
        action_feat = self.action_encoder(action)           # (B, embed_dim)
        instr_feat = self.encode_instruction(instruction_list)  # (B, embed_dim)
        combined = torch.cat([state_feat, action_feat, instr_feat], dim=1)
        return self.cfn(combined)  # 输出 shape: (B, cfn_output_dim)

    def compute_loss(self, state, action, instruction_list, coin_flip_targets):
        preds = self.forward(state, action, instruction_list)
        loss = F.mse_loss(preds, coin_flip_targets)
        return loss, preds
