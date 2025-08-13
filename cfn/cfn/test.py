import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.models.idefics3.image_processing_idefics3 import Idefics3ImageProcessor
from cfn_dataset import resize_torch
from torchvision import transforms

# 设置设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 随机生成一张 RGB 图像（224x224）
img = Image.open('/gemini/space/users/ysy/project/vla_post_train/cook_baozi_swapped.png').convert('RGB')

transform = transforms.ToTensor()
image = transform(img)
# import ipdb; ipdb.set_trace()


# 加载模型和处理器
model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
processor.image_processor.do_image_splitting = False
processor.image_processor.do_rescale = False
processor.image_processor.do_resize = False
processor.image_processor.do_convert_rgb = False
# # import ipdb; ipdb.set_trace()
Idefics3ImageProcessor.resize = resize_torch


model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    # 改为默认方式
    _attn_implementation="eager",
).to(DEVICE)

# model = AutoModelForVision2Seq.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
#     _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
# ).to(DEVICE)

# 构造对话 prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the scene."}
        ]
    },
]

# 编码输入
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
import ipdb; ipdb.set_trace()
# breakpoint()
# 生成输出
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=256)

# 解码
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("🧠 Output:", output)
