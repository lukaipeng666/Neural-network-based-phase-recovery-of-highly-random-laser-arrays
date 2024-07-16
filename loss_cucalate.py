import numpy as np
import cv2
from scipy.fft import ifft2, ifftshift
import os
import matplotlib.pyplot as plt

path1 = './test_out_gs_1'
path2 = 'test_out_neu'
path3 = 'test_in'
path = 'temp'

for imgs in os.listdir(path2):
    img_path = os.path.join(path2, imgs)
    G0_GSNewW = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if G0_GSNewW is None:
        raise ValueError("Could not open or find the image.")

    G0_GSNewW = G0_GSNewW.astype(np.float64) / 255.0  # 归一化到0-1范围

    # 将强度转化成-pi到pi范围
    G0_GSNewW = (G0_GSNewW * 2 - 1) * np.pi

    # 转化为虚数表示
    G0_GSNewW = np.exp(1j * G0_GSNewW)

    # 进行傅里叶逆变换
    g0_GSNewB = ifft2(ifftshift(G0_GSNewW))

    # 归一化模到 0-255 范围
    amplitude = np.abs(g0_GSNewB)
    mean_img = np.mean(amplitude)
    factor = 5 / mean_img
    normalized_amplitude = amplitude * factor

    filement_name = './xxxm.png'
    cv2.imwrite(filement_name, normalized_amplitude)

    path_ori_img = os.path.join(path3, imgs)
    img_ori = cv2.imread(path_ori_img, cv2.IMREAD_GRAYSCALE)
    var = np.mean(np.square(img_ori - normalized_amplitude))
    print(var)




