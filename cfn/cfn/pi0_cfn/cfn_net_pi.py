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



# ========== CFN 包装类（支持图文编码） ========== #
class CFNWrapper_pi(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 vision_model_name='HuggingFaceTB/SmolVLM-500M-Instruct',
                 embed_dim=128,
                 cfn_output_dim=64,
                 cfn_action_steps=None,
                 pretrained_checkpoint_path=None
                 ):
        super().__init__()

        print(f"loading pretrained checkpoint from {pretrained_checkpoint_path}...")
        self.policy = PI0Policy.from_pretrained(pretrained_checkpoint_path, local_files_only=True).eval()
        print("loading model success!")

        import ipdb; ipdb.set_trace()

        # self.vision_model.eval()
        # for p in self.vision_model.parameters():
        #     p.requires_grad = False

        # hidden_size = self.vision_model.config.text_config.hidden_size
        # self.language_proj = nn.Linear(hidden_size, embed_dim)

        # 多模态融合预测模块
        total_feature_dim = embed_dim * 3
        self.cfn = CoinFlippingNetwork(input_dim=total_feature_dim,
                                       output_dim=cfn_output_dim)

        assert cfn_action_steps!=None, "cfn_action_steps can not be None !"
        self.cfn_action_steps = cfn_action_steps



    def extract_images_from_batch(self, batch):
        """
        从 batch 中提取 3 张图像并转为 PIL 格式
        返回: List[List[PIL.Image]]，每个样本含 3 张图
        """
        cam_high = batch['observation.images.cam_high']
        cam_left = batch['observation.images.cam_left_wrist']
        cam_right = batch['observation.images.cam_right_wrist']
        B = cam_high.shape[0]
        # import ipdb; ipdb.set_trace()
        images = []
        for i in range(B):
            imgs = [
                self.to_pil(cam_high[i].cpu()),
                self.to_pil(cam_left[i].cpu()),
                self.to_pil(cam_right[i].cpu())
                # cam_high[i],
                # cam_left[i],
                # cam_right[i]
            ]
            images.append(imgs)
        
        # import ipdb; ipdb.set_trace()
        return images

    def extract_instructions(self, batch):
        return batch['task']  # 应该是 List[str]

    def encode_instruction(self, image_list, text_list):
        # 构建 chat prompt（每个样本含 3 张图）
        messages_batch = []
        for text in text_list:
            messages_batch.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]
            })

        # breakpoint()
        prompts = [self.processor.apply_chat_template([m], add_generation_prompt=True)
                   for m in messages_batch]

        inputs = self.processor(
            text=prompts,
            images=image_list,  # List[List[PIL.Image]]
            return_tensors="pt",
            padding=True, 
            do_resize=False,
        ).to(next(self.parameters()).device)

        
        # from torchvision.transforms.functional import to_pil_image
        # img_tensor = inputs['pixel_values'][0, 0]  # shape: [3, 512, 512]
        # # 确保在 CPU 上
        # img_tensor = img_tensor.detach().cpu()
        # # 将张量转换为 PIL 图像
        # img = to_pil_image(img_tensor)
        # # 保存图像到磁盘
        # img.save("restored_image_2_0726.png")

        # import ipdb; ipdb.set_trace()

        # breakpoint()
        with torch.no_grad():
            outputs = self.vision_model(**inputs, output_hidden_states=True)
            # outputs = self.vision_model(**inputs)
            # print('good')
            # breakpoint()
            last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]

        cls_embedding = last_hidden[:, 0, :]  # or mean-pooling
        return self.language_proj(cls_embedding)  # [B, embed_dim]

    def encode_instruction2(self, smol_inputs):
        with torch.no_grad():
            outputs = self.vision_model(**smol_inputs, output_hidden_states=True)
            # outputs = self.vision_model(**inputs)
            # print('good')
            # breakpoint()
            last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]

        cls_embedding = last_hidden[:, 0, :]  # or mean-pooling
        return self.language_proj(cls_embedding)  # [B, embed_dim]


    def forward(self, batch):
        # import ipdb; ipdb.set_trace()
        # breakpoint()
        state = batch['observation.state'].to(next(self.parameters()).device)
        action = batch['action'].to(next(self.parameters()).device)
        smol_inputs = {key: value.to(next(self.parameters()).device) for key, value in batch['smol_inputs'].items()}

        action = action[:, :self.cfn_action_steps, :]
        # batch_size = action.shape[0]
        # action = action.reshape(batch_size, -1)

        # breakpoint()
        # images = self.extract_images_from_batch(batch)
        # instruction_list = self.extract_instructions(batch)
        # breakpoint()

        state_feat = self.state_encoder(state)
        action_feat, _ = self.action_encoder(action)
        
        # import ipdb; ipdb.set_trace()
        # instr_feat = self.encode_instruction(images, instruction_list)
        instr_feat = self.encode_instruction2(smol_inputs)
        # breakpoint()
        # instr_feat = self.encode_instruction(batch['observation.images.cam_high'], instruction_list)
        # import ipdb; ipdb.set_trace()
        combined = torch.cat([state_feat, action_feat, instr_feat], dim=1)
        return self.cfn(combined)

    def compute_loss(self, batch):
        preds = self.forward(batch)
        targets = batch["CoinFlip_target"].float().to(next(self.parameters()).device)
        loss = F.mse_loss(preds, targets)
        return loss, preds
