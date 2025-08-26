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

class cfn_sub_net(nn.Module):
    def __init__(self, img_lang_feature_dim, cfn_action_steps, action_dim, embed_dim, output_dim=20):
        super().__init__()
        self.proj = nn.Linear(img_lang_feature_dim, embed_dim)

        # 多模态融合预测模块
        self.cfn = CoinFlippingNetwork(
            input_dim=embed_dim*2,
            output_dim=output_dim
        )

        # self.output = nn.Linear(20, output_dim)
        perceiver_cfg = PerceiverConfig(
            dim = embed_dim,
            latent_dim = 512,
            num_latents = 32,  # 256
            depth = 2,
            cross_heads = 1,
            cross_dim_head = 64,
            latent_heads = 8,
            latent_dim_head = 64,
            attn_dropout = 0.,  # 0.1
            ff_dropout = 0.,    # 0.1
            output_dim = embed_dim,
            final_proj_head = True,
        )

        self.action_encoder = SequentialActionEmb(
            num_agents=1,
            num_steps_conditioning=cfn_action_steps,
            action_dim=action_dim,
            is_continuous_act=1,
            perceiver_cfg=perceiver_cfg,
        )

    def forward(self, img_lang_feature, action_chunk):
        img_lang_emb = self.proj(img_lang_feature)
        action_emb, _ = self.action_encoder(action_chunk)
        
        combined = torch.cat([img_lang_emb, action_emb], dim=1)
        import ipdb; ipdb.set_trace()
        return self.cfn(combined)



# ========== CFN 包装类（支持图文编码） ========== #
class CFNWrapper_pi_sub(nn.Module):
    def __init__(self,
                 cfn_output_dim=20,
                 cfn_action_steps=25,
                 pretrained_checkpoint_path=None,
                 embed_dim=512,
                 action_dim=14, # 每个action的dim
                 pick_num = 2,
                 ):
        super().__init__()
        # self.cut_num = 2
        self.pick_num = pick_num
        self.cfn_action_steps = cfn_action_steps

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

        self.cfn_sub = cfn_sub_net(
            img_lang_feature_dim = 2048, 
            cfn_action_steps = cfn_action_steps, 
            action_dim = action_dim, 
            embed_dim = embed_dim, 
            output_dim=cfn_output_dim,
        ).to(next(self.policy.parameters()).device)



    def forward(self, batch):
        # import ipdb; ipdb.set_trace()
        # breakpoint()

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(next(self.parameters()).device)

        with torch.no_grad():
            features = self.policy.get_im_lang_feature(batch)
            features = features.to(next(self.parameters()).dtype)
        # import ipdb; ipdb.set_trace()
        index = [self.cfn_action_steps * (self.pick_num-1), self.cfn_action_steps * (self.pick_num)]
        action_chunk = batch['action'][:, index[0] : index[1], :]
        # import ipdb; ipdb.set_trace()
        return self.cfn_sub(features, action_chunk)

    def compute_loss(self, batch):
        preds = self.forward(batch)
        targets = batch["CoinFlip_target"].float().to(next(self.parameters()).device)
        loss = F.mse_loss(preds, targets)
        return loss, preds
