import random
import os
import torch
import torchvision.transforms as transform
from PIL import Image, ImageDraw
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from split_my_data import split_data_real


image_size = 256
spot_radius = 2
itera = 5
seed_value = 30
global G0_GSNew


def normalize_to_minus1_1(tensor):
    return tensor * 2.0 - 1.0


transform = transform.Compose([
    transform.Resize((256, 256)),
    transform.ToTensor(),
    normalize_to_minus1_1
])


def data_generate_seed():
    inpus_list = []
    output_list = []
    output_list_10 = []
    for first_x in range(6, 140, 8):
        for first_y in range(6, 140, 8):
            num = 0
            while num < 3:
                image = Image.new('L', (image_size, image_size), 0)
                for x in range(first_x, first_x + 49, 12):
                    for y in range(first_y, first_y + 49, 12):
                        draw = ImageDraw.Draw(image)
                        random.seed(None)
                        x_radom = random.randrange(x - 4, x + 5, 4)
                        y_radom = random.randrange(y - 4, y + 5, 4)
                        draw.ellipse(
                            (
                                x_radom - spot_radius, y_radom - spot_radius, x_radom + spot_radius,
                                y_radom + spot_radius),
                            fill=255)  # 255 表示白色
                # filename = f"spot_at_{first_x}_{first_y}_{num}.png"
                # path1 = './test_in'
                # path2 = os.path.join(path1, filename)
                # image.save(path2)
                image2 = np.asarray(image)
                image2 = image2.astype(np.float32) / 255.0

                output_list.append(gs(image2, first_x, first_y, num)[0])
                output_list_10.append(gs(image2, first_x, first_y, num)[1])

                image2 = torch.from_numpy(image2)
                if image2.dtype == torch.float64:
                    image2 = image2.float()
                normalized_tensor_image = image2 * 2 - 1
                inpus_list.append(normalized_tensor_image)
                num += 1
    train_ori_imglist, train_ori_imglist_out, val_ori_imglist, val_ori_imglist_out = split_data_real(inpus_list,
                                                                                                     output_list,
                                                                                                     output_list_10)
    return train_ori_imglist, train_ori_imglist_out, val_ori_imglist, val_ori_imglist_out


def gs(image, first_x, first_y, num):
    global G0_GSNew, phase_image_normalized_10


    Amplitude = cv2.resize(image, (256, 256))
    Amplitude = Amplitude / Amplitude.max()


    np.random.seed(seed_value)
    phase = np.random.rand(*Amplitude.shape) * 2 * np.pi
    np.random.seed(None)

    g0_GS = Amplitude * np.exp(1j * phase)

    for n in range(itera):
        G0_GS = fftshift(fft2(g0_GS))  
        G0_GSNew = G0_GS / np.abs(G0_GS) 
        g0_GSNew = ifft2(ifftshift(G0_GSNew)) 
        g0_GS = Amplitude * (
                g0_GSNew / np.abs(g0_GSNew)) 
        if n == 1:
            phase_image_10 = np.angle(G0_GSNew)
            phase_image_normalized_10 = (phase_image_10 + np.pi) / (2 * np.pi)

            # outputFileName = f"outputFileName = spot_at_{first_x}_{first_y}_{num}.png"
            # path1 = './test_out_2'
            # outputFileName = os.path.join(path1, outputFileName)
            # phase_image_normalized_save = (phase_image_10 + np.pi) / (
            #             2 * np.pi)  # Normalize the phase to [0, 1] range
            # cv2.imwrite(outputFileName, phase_image_normalized_save * 255) 

            phase_image_normalized_10 = torch.from_numpy(phase_image_normalized_10)
            phase_image_normalized_10 = phase_image_normalized_10 * 2 - 1

    phase_image = np.angle(G0_GSNew)
    phase_image_normalized = (phase_image + np.pi) / (2 * np.pi)

    # outputFileName = f"outputFileName = spot_at_{first_x}_{first_y}_{num}.png"
    # path1 = './test_out'
    # outputFileName = os.path.join(path1, outputFileName)
    # phase_image_normalized_save = (phase_image + np.pi) / (2 * np.pi)
    # cv2.imwrite(outputFileName, phase_image_normalized_save * 255)

    phase_image_normalized = torch.from_numpy(phase_image_normalized)
    phase_image_normalized = phase_image_normalized * 2 - 1
    if phase_image_normalized.dtype == torch.float64:
        phase_image_normalized = phase_image_normalized.float()
    if phase_image_normalized_10.dtype == torch.float64:
        phase_image_normalized_10 = phase_image_normalized_10.float()
    return phase_image_normalized, phase_image_normalized_10


