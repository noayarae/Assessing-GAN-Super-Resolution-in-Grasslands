
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers, Model
from sklearn.model_selection import train_test_split

import numpy as np
from keras import Model
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten, Resizing, Lambda
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add, GlobalAveragePooling2D
from tqdm import tqdm

from keras.applications import VGG19

from tifffile import imread
import random
# -----------------------------------------------------------------------------



# Load first 80 images from the dataset
n = 100

#lr_list = os.listdir('dataset/lr_imgs_100')[:n]
#hr_list = os.listdir('dataset/hr_imgs_100')[:n]
#lr_list = os.listdir('/content/drive/SharedWithMe/downscaling_data/lr_imgs_100')
#lr_list = os.listdir('/content/drive/SharedWithMe/downscaling_data/hr_imgs_100')

#lr_list = os.listdir('/content/drive/My Drive/5000_data/downscaling_data/lr_imgs_100')[:n]
#hr_list = os.listdir('/content/drive/My Drive/5000_data/downscaling_data/hr_imgs_100')[:n]

#lr_list = os.listdir('D:/work/research_t/downscaling/put_all_imgs/set3_hn/lr_imgs_100')[:n]
#hr_list = os.listdir('D:/work/research_t/downscaling/put_all_imgs/set3_hn/hr_imgs_100')[:n]

lr_list = [f for f in os.listdir(r'D:/work/research_t/downscaling/put_all_imgs/set3_hn/lr_imgs_100')
           if f.lower().endswith('.tif')][:n]
hr_list = [f for f in os.listdir(r'D:/work/research_t/downscaling/put_all_imgs/set3_hn/hr_imgs_100')
           if f.lower().endswith('.tif')][:n]


lr_images = []
for img in lr_list:
    img_lr = imread('D:/work/research_t/downscaling/put_all_imgs/set3_hn/lr_imgs_100/' + img)
    lr_images.append(img_lr)
lr_images = np.array(lr_images)

hr_images = []
for img in hr_list:
    img_hr = imread('D:/work/research_t/downscaling/put_all_imgs/set3_hn/hr_imgs_100/' + img)
    hr_images.append(img_hr)
hr_images = np.array(hr_images)



# View few images
image_number = random.randint(0, len(lr_images)-1)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(lr_images[image_number][:,:,0])
plt.title('Low Resolution Image')
plt.subplot(1,2,2)
plt.imshow(hr_images[image_number][:,:,0])
plt.title('High Resolution Image')
plt.show()


# scale values
image_number = random.randint(0, len(lr_images)-1)
# channel 0
min_val = np.min(lr_images[image_number][:,:,0])
max_val = np.max(lr_images[image_number][:,:,0])
print(f"Channel 0 Min value: {min_val}, Max value: {max_val}")
# channel 1
min_val = np.min(lr_images[image_number][:,:,1])
max_val = np.max(lr_images[image_number][:,:,1])
print(f"Channel 1 Min value: {min_val}, Max value: {max_val}")
# channel 2
min_val = np.min(lr_images[image_number][:,:,2])
max_val = np.max(lr_images[image_number][:,:,2])
print(f"Channel 2 Min value: {min_val}, Max value: {max_val}")
# channel 3
min_val = np.min(lr_images[image_number][:,:,3])
max_val = np.max(lr_images[image_number][:,:,3])
print(f"Channel 3 Min value: {min_val}, Max value: {max_val}")



# LR: Normalize the images for each channel independently
min_vals = lr_images.min(axis=(0, 1, 2))
max_vals = lr_images.max(axis=(0, 1, 2))
print("Per-channel min:", min_vals)
print("Per-channel max:", max_vals)
lr_images_norm = (lr_images - min_vals) / (max_vals - min_vals)



# HR: Normalize the images for each channel independently
min_vals = hr_images.min(axis=(0, 1, 2))
max_vals = hr_images.max(axis=(0, 1, 2))
print("Per-channel min:", min_vals)
print("Per-channel max:", max_vals)
hr_images_norm = (hr_images - min_vals) / (max_vals - min_vals)



# View few images
image_number = random.randint(0, len(lr_images)-1)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(lr_images_norm[image_number][:,:,0])
plt.title('Low Resolution Image')
plt.subplot(1,2,2)
plt.imshow(hr_images_norm[image_number][:,:,0])
plt.title('High Resolution Image')
plt.show()




lr_images_norm = lr_images_norm[..., :3]
hr_images_norm = hr_images_norm[..., :3]
print(lr_images_norm.shape)
print(hr_images_norm.shape)


lr_images_norm_32 = np.array([cv2.resize(img, (32, 32)) for img in lr_images_norm])
print(lr_images_norm_32.shape)
# (100, 32, 32, 3)

hr_images_norm_128 = np.array([cv2.resize(img, (128, 128)) for img in hr_images_norm])
print(hr_images_norm_128.shape)
# (100, 128, 128, 3)
# hr_images_norm = resized_batch

#np.save('D:/work/research_t/downscaling/put_all_imgs/set3_hn/lr_images_norm_32.npy', lr_images_norm_32)
#np.save('D:/work/research_t/downscaling/put_all_imgs/set3_hn/hr_images_norm_128.npy', hr_images_norm_128)




















