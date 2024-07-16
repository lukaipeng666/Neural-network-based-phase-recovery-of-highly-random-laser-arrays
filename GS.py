import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

itera = 120


import os
import glob

imageFiles = glob.glob('*.png')

for fileIdx, fileName in enumerate(imageFiles):

    gray = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64) 

    Amplitude = cv2.resize(gray, (256, 256))
    Amplitude = Amplitude / Amplitude.max()
    # Generate a random phase
    np.random.seed(30)
    phase = np.random.rand(*Amplitude.shape) * 2 * np.pi

    g0_GS = Amplitude * np.exp(1j * phase)

    for n in range(itera):
        G0_GS = fftshift(fft2(g0_GS)) 
        G0_GSNew = G0_GS / np.abs(G0_GS)
        g0_GSNew = ifft2(ifftshift(G0_GSNew))
        g0_GS = Amplitude * (
                    g0_GSNew / np.abs(g0_GSNew))

    baseName, _ = os.path.splitext(os.path.basename(fileName))
    outputFileName = f"{baseName}_N.png"
    phase_image = np.angle(G0_GSNew)
    phase_image_normalized = (phase_image + np.pi) / (2 * np.pi) 
    cv2.imwrite(outputFileName, phase_image_normalized * 255)
