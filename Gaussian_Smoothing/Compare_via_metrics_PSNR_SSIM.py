import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True)

# Read images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Calculate PSNR and SSIM
psnr_value = calculate_psnr(img1, img2)
ssim_value = calculate_ssim(img1, img2)

print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")