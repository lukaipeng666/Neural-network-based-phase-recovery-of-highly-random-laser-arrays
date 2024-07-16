from net import unet
import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def great(img_path):
    # 使用PIL读取图像，并确保它是灰度图
    img = Image.open(img_path).convert('L')  # 转换为灰度图

    transforms = transform.Compose([
        transform.Resize((256, 256)),
        transform.ToTensor()
    ])
    img = transforms(img)
    img = img[None].to('cuda')  # [1, 1, 256, 256] 注意这里通道数为1了

    # 实例化网络并加载权重（确保权重与模型匹配）
    G = unet().to('cuda')
    ckpt = torch.load('weights/pix2pix_256.pth')
    # 如果权重不匹配，这里可能会出问题。确保ckpt['G_model']的键和结构与你的模型匹配。
    G.load_state_dict(ckpt['G_model'], strict=False)
    G.eval()

    with torch.no_grad():
        out = G(img)[0]
        out = out.permute(1, 2, 0)  # 如果输出是单通道的，这里可能不需要这个permute操作，具体取决于你的unet实现。
        out = out.cpu().detach().numpy()  # 将范围从 [-1, 1] 转换为 [0, 1]（如果模型输出是在这个范围内的话）

    # 处理输出图像（假设输出是单通道的灰度图）
    out_uint8 = (out * 255).astype(np.uint8)  # 将浮点数数组转换为整数数组
    image = Image.fromarray(out_uint8.squeeze())  # squeeze掉单通道维度（如果存在的话）
    image.save('spot_at_128_128_holo.png')  # 保存输出图像为PNG格式（或其他格式）

    # 可选：显示图像（调试用）
    plt.imshow(out_uint8.squeeze(), cmap='gray')  # 使用灰度色图显示单通道图像
    plt.axis('off')  # 不显示坐标轴
    plt.show()


if __name__ == '__main__':
    great('./base/spot_at_128_128.jpg')  # 确保路径和文件名正确无误（且文件存在）