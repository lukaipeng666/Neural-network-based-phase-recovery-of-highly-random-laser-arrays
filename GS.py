import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

itera = 120

# Get a list of all PNG image files in the current directory
import os
import glob

imageFiles = glob.glob('*.png')

for fileIdx, fileName in enumerate(imageFiles):
    # Read the image and convert it to grayscale
    gray = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    gray = gray.astype(np.float64)  # Convert to double precision for calculations

    # Resize and normalize the image
    Amplitude = cv2.resize(gray, (256, 256))
    Amplitude = Amplitude / Amplitude.max()  # Normalize to [0, 1] range

    # Generate a random phase
    np.random.seed(30)
    phase = np.random.rand(*Amplitude.shape) * 2 * np.pi

    # Create the initial complex amplitude distribution for the GS algorithm
    g0_GS = Amplitude * np.exp(1j * phase)

    # Perform the iterative process
    for n in range(itera):
        G0_GS = fftshift(fft2(g0_GS))  # Fourier transform to frequency domain
        G0_GSNew = G0_GS / np.abs(G0_GS)  # Take the phase value, frequency domain with full 1 amplitude constraint
        g0_GSNew = ifft2(ifftshift(G0_GSNew))  # Inverse Fourier transform back to spatial domain
        g0_GS = Amplitude * (
                    g0_GSNew / np.abs(g0_GSNew))  # Directly use the initial amplitude constraint without modification

    # Save the final phase image as a JPG file
    baseName, _ = os.path.splitext(os.path.basename(fileName))  # Remove the .png extension
    outputFileName = f"{baseName}_N.png"
    phase_image = np.angle(G0_GSNew)  # Extract the phase information
    phase_image_normalized = (phase_image + np.pi) / (2 * np.pi)  # Normalize the phase to [0, 1] range
    cv2.imwrite(outputFileName, phase_image_normalized * 255)  # Save the phase image as a JPG file