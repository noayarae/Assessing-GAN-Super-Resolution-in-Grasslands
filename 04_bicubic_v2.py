# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:10:02 2025

@author: efrain.noa-yarasca
"""

import numpy as np
import cv2, os

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

#from sklearn.model_selection import train_test_split
#from tensorflow.keras.applications import VGG16
#from tensorflow.keras.applications.vgg16 import preprocess_input

import random
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'



# Load .npy files
#lr_data = np.load('Entrop_q4_sat_imgs_norm_64x64_4b.npy')    # (1286, 64, 64, 4)
#hr_data = np.load('Entrop_q4_drn_imgs_norm_256x256_4b.npy')  # (1286, 256, 256, 4)
#   lr_data = np.load(r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_entropy\Entrop_q4_drn_imgs_norm_64x64_4b.npy")
#   hr_data = np.load(r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_entropy\Entrop_q4_drn_imgs_norm_256x256_4b.npy")
#"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_Text\Text_C0_drn_imgs_norm_64x64_4b.npy"

#lr_data = np.load(r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_ndvi\NDVI_q4_sat_imgs_norm_64x64_4b.npy")
#hr_data = np.load(r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_ndvi\NDVI_q4_drn_imgs_norm_256x256_4b.npy")


lr_data = np.load(r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_ndvi\NDVI_q1_sat_imgs_norm_64x64_4b.npy")
hr_data = np.load(r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_ndvi\NDVI_q1_drn_imgs_norm_256x256_4b.npy")
#lr_data = np.load(r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_ndvi\NDVI_q1_drn_imgs_norm_64x64_4b.npy")



print("LR & HR shape", lr_data.shape, hr_data.shape)

# Use only first 3 channels
lr_data = lr_data[:, :, :, :3]  # (1286, 64, 64, 3) # lr_images[:511, :, :, :3] # lr_data[:904, :, :, :3]
hr_data = hr_data[:, :, :, :3]  # (1286, 256, 256, 3) # hr_images[:511, :, :, :3] # hr_data[:904, :, :, :3] 
print("LR & HR shape", lr_data.shape, hr_data.shape)



# Split into train/test (80% train, 20% test)
#lr_train, lr_test, hr_train, hr_test = train_test_split(lr_data, hr_data, test_size=0.2, random_state=42)
lr_test, hr_test = lr_data, hr_data

# Upscale LR test tiles using bicubic interpolation
bicubic_upscaled = []
for img in tqdm(lr_test, desc="Upscaling with Bicubic Interpolation"):
    upscaled_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    bicubic_upscaled.append(upscaled_img)

bicubic_upscaled = np.array(bicubic_upscaled)  # (257, 256, 256, 3)



# Flatten images for MAE/MSE comparison
def flatten_images(imgs):
    return imgs.reshape((imgs.shape[0], -1))

# Compute metrics
psnr_vals = []
ssim_vals = []
mae_vals = []
mse_vals = []
lpips_vals = []

for i in tqdm(range(len(bicubic_upscaled)), desc="Evaluating Metrics"):
    pred = bicubic_upscaled[i]
    true = hr_test[i]

    psnr_vals.append(psnr(true, pred, data_range=1.0))
    ssim_vals.append(ssim(true, pred, channel_axis=-1, data_range=1.0))

    pred_flat = flatten_images(pred)
    true_flat = flatten_images(true)
    mae_vals.append(mean_absolute_error(true_flat, pred_flat))
    mse_vals.append(mean_squared_error(true_flat, pred_flat))
    



# Show average metrics on test set
'''
print("=== Bicubic Interpolation Performance (Test Set) ===")
print(f"PSNR: {np.mean(psnr_vals):.2f} dB")
print(f"SSIM: {np.mean(ssim_vals):.4f}")
print(f"MAE : {np.mean(mae_vals):.6f}")
print(f"MSE : {np.mean(mse_vals):.6f}") #'''
#print(f"Perceptual (VGG): {perceptual_score:.6f}")

# Compute mean and standard deviation for all metrics
mean_psnr = np.mean(psnr_vals)
sd_psnr = np.std(psnr_vals)
mean_ssim = np.mean(ssim_vals)
sd_ssim = np.std(ssim_vals)
mean_mae = np.mean(mae_vals)
sd_mae = np.std(mae_vals)
mean_mse = np.mean(mse_vals)
sd_mse = np.std(mse_vals)


# Print results
print(f"\nBicubic Model (Quartile 4) Metrics:")
print(f"PSNR: Mean = {mean_psnr:.4f}, SD = {sd_psnr:.4f}")
print(f"SSIM: Mean = {mean_ssim:.4f}, SD = {sd_ssim:.4f}")
print(f"MAE: Mean = {mean_mae:.4f}, SD = {sd_mae:.4f}")
print(f"MSE: Mean = {mean_mse:.4f}, SD = {sd_mse:.4f}")





def stretch_image(img):
    # img: (H,W,3) normalized [0–1] or not
    out = np.zeros_like(img)
    for i in range(3):
        p2, p98 = np.percentile(img[:,:,i], (2, 98))
        out[:,:,i] = np.clip((img[:,:,i] - p2) / (p98 - p2 + 1e-6), 0, 1)
    return out



# -------------------------------
# Select a representative sample
# -------------------------------
idx = 0   # you can change this index (e.g., 10, 50, 100)
#idx = np.argmin(np.abs(psnr_vals - np.median(psnr_vals)))

lr_sample = lr_test[idx]              # (64, 64, 3)
bicubic_sample = bicubic_upscaled[idx]  # (256, 256, 3)
hr_sample = hr_test[idx]              # (256, 256, 3)

# -------------------------------
# Plot LR → Bicubic → HR
# -------------------------------
plt.figure(figsize=(12, 4))

# LR
plt.subplot(1, 3, 1)
plt.imshow(stretch_image(lr_sample))
plt.title("LR Input (64×64)")
plt.axis("off")

# Bicubic Prediction
plt.subplot(1, 3, 2)
plt.imshow(stretch_image(bicubic_sample))
plt.title("Bicubic Upscaled (256×256)")
plt.axis("off")

# HR Target
plt.subplot(1, 3, 3)
plt.imshow(stretch_image(hr_sample))
plt.title("HR Target (256×256)")
plt.axis("off")

plt.tight_layout()
plt.show()



#lr_data = np.load(r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_ndvi\NDVI_q1_sat_imgs_norm_64x64_4b.npy")


'''
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

psnr_val = psnr(hr_sample, bicubic_sample, data_range=1.0)
ssim_val = ssim(hr_sample, bicubic_sample, channel_axis=-1, data_range=1.0)

plt.subplot(1, 3, 2)
plt.imshow(bicubic_sample)
plt.title(f"Bicubic\nPSNR={psnr_val:.2f} dB | SSIM={ssim_val:.3f}")
plt.axis("off")
#'''




























