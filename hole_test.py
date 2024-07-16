from net import unet
import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
def great(img_path):
    if img_path.endswith('.png'):
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]
    else:
        img = Image.open(img_path)

    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transforms(img.copy())
    img = img[None].to('cuda')  # [1,3,128,128]

    # 实例化网络
    G = unet().to('cuda')
    # 加载预训练权重
    ckpt = torch.load('weights/pix2pix_256.pth')
    G.load_state_dict(ckpt['G_model'], strict=False)

    G.eval()
    out = G(img)[0]
    out = out.permute(1,2,0)
    out = (0.5 * (out + 1)).cpu().detach().numpy()
    plt.figure()
    plt.imshow(out)
    plt.show()
    # 将浮点数数组的范围从 [0, 1] 转换为 [0, 255] 的整数
    out_uint8 = (out * 255).astype(np.uint8)

    # 创建一个PIL图像对象
    image = Image.fromarray(out_uint8)

    # 保存为JPG
    image.save('output_spot_at_speckled_image_66.png')


if __name__ == '__main__':
    great('./speckled_image_66.jpg')
