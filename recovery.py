import numpy as np
import cv2
from scipy.fft import ifft2, ifftshift
import matplotlib.pyplot as plt
import torch
import torch.fft as fft

# G0_GSNewW = cv2.imread('output_image.png', cv2.IMREAD_GRAYSCALE)
# if G0_GSNewW is None:
#     raise ValueError("Could not open or find the image.")
#
# G0_GSNewW = G0_GSNewW.astype(np.float64) / 255.0  # 归一化到0-1范围
#
# # 将强度转化成-pi到pi范围
# G0_GSNewW = (G0_GSNewW * 2 - 1) * np.pi
#
# # 转化为虚数表示
# G0_GSNewW = np.exp(1j * G0_GSNewW)
#
# # 进行傅里叶逆变换
# g0_GSNewB = ifft2(ifftshift(G0_GSNewW))
#
# # 归一化模到 0-255 范围
# normalized_amplitude = (np.abs(g0_GSNewB) - np.min(np.abs(g0_GSNewB))) / (
#         np.max(np.abs(g0_GSNewB)) - np.min(np.abs(g0_GSNewB))) * 255
# file_name = 'spot_at_128_128.png'
# G0_GSNewW = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
# if G0_GSNewW is None:
#     raise ValueError(f"Could not open or find the image: {file_name}")
# G0_GSNewW = G0_GSNewW.astype(np.float64) / 255.0  # 归一化到0-1范围
# G0_GSNewW = (G0_GSNewW * 2 - 1) * np.pi  # 将强度转化成-pi到pi范围
# G0_GSNewW = np.exp(1j * G0_GSNewW)  # 转化为虚数表示
# g0_GSNewB = ifft2(ifftshift(G0_GSNewW))  # 进行傅里叶逆变换
# normalized_amplitude = (np.abs(g0_GSNewB) - np.min(np.abs(g0_GSNewB))) / (
#         np.max(np.abs(g0_GSNewB)) - np.min(np.abs(g0_GSNewB))) * 255 # 归一化模到 0-255 范围
#
# # 显示图像（确保是灰度图）
# plt.imshow(normalized_amplitude.astype(np.uint8), cmap='gray')
# plt.axis('off')
# plt.show()
#
# # 保存图像
# cv2.imwrite('spot_at_128_128_holo_GS_recovery.png', normalized_amplitude.astype(np.uint8))
file_name = './test_out_gs_10/outputFileName = spot_at_26_6_0.png'
G0_GSNewW = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
if G0_GSNewW is None:
    raise ValueError(f"Could not open or find the image: {file_name}")

# 将 NumPy 数组转换为 PyTorch 张量
G0_GSNewW = torch.from_numpy(G0_GSNewW).to(torch.float64) / 255.0  # 归一化到 0-1 范围
G0_GSNewW = (G0_GSNewW * 2 - 1) * torch.pi  # 将强度转化为 -pi 到 pi 范围

G0_GSNewW_complex = torch.complex(torch.zeros_like(G0_GSNewW), G0_GSNewW)  # 转化为虚数表示

G0_GSNewW_shifted = fft.ifftshift(G0_GSNewW_complex)
g0_GSNewB = fft.ifft2(G0_GSNewW_shifted)

# 计算归一化幅度
normalized_amplitude = (torch.abs(g0_GSNewB) - torch.min(torch.abs(g0_GSNewB))) / (
        torch.max(torch.abs(g0_GSNewB)) - torch.min(torch.abs(g0_GSNewB))) * 255  # 归一化模到 0-255 范围

# 转换回 NumPy 数组以便显示
normalized_amplitude_np = normalized_amplitude.detach().numpy().astype(np.uint8)

# # 显示图像（确保是灰度图）
# plt.imshow(normalized_amplitude_np, cmap='gray')
# plt.axis('off')
# plt.show()

# 保存图像
