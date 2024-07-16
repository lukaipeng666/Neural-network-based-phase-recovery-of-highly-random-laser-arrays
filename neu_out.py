from net import U_net as unet
import torch
import torchvision.transforms as transform
from PIL import Image
import numpy as np
import os
import time


def normalize_to_minus1_1(Tensor):
    return Tensor * 2 - 1


transforms = transform.Compose([
    transform.Resize((256, 256)),
    transform.ToTensor(),
    normalize_to_minus1_1
])

# 实例化网络并加载权重（确保权重与模型匹配）
G = unet().to('cuda')
ckpt = torch.load('weights/Bi_Ushape_Attendion.pth')
# 如果权重不匹配，这里可能会出问题。确保ckpt['G_model']的键和结构与你的模型匹配。
G.load_state_dict(ckpt['G_model'], strict=False)


def great(input_img_path, output_dir, output_basename):
    # 使用PIL读取图像，并确保它是灰度图
    img = Image.open(input_img_path).convert('L')  # 转换为灰度图
    G.eval()
    img = transforms(img)
    img = img
    img = img[None].to('cuda')  # [1, 1, 256, 256] 注意这里通道数为1了

    with torch.no_grad():
        out = G(img)[0]
        out = (out + 1) * 0.5
        out = out.cpu().detach().numpy()  # 将范围从 [-1, 1] 转换为 [0, 1]（如果模型输出是在这个范围内的话）

    out_uint8 = (out * 255).astype(np.uint8)  # 将浮点数数组转换为整数数组
    output_image = Image.fromarray(out_uint8.squeeze())  # squeeze掉单通道维度（如果存在的话）

    # 构建输出文件的完整路径
    output_filename = f"{output_basename}.png"  # 或者其他你想要的命名方式
    output_path = os.path.join(output_dir, output_filename)

    # 保存输出图像为PNG格式
    output_image.save(output_path)


input_dir = './test_in'
output_dir = './test_out_neu'

# 遍历输入目录中的所有图像文件
all_images = os.listdir(input_dir)
# a = time.time()
num = 0
for image_filename in all_images:

    if image_filename.lower().endswith('.png'):  # 确保只处理图像文件
        input_img_path = os.path.join(input_dir, image_filename)
        basename, _ = os.path.splitext(image_filename)
        great(input_img_path, output_dir, basename)
        print('converting')

# b = time.time()
# print('总时间为：{}sec，平均时间为：{}sec'.format(b-a,(b-a) / 200))
