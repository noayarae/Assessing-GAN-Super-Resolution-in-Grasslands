"""
Created on Wed Jun 18 11:23:25 2025
@author: efrain.noa-yarasca
environment: tf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, Model
from sklearn.model_selection import train_test_split

#from keras import Model
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten, Concatenate
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from skimage.metrics import structural_similarity as compare_ssim
from keras.optimizers import Adam
from keras.applications import VGG19
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import skimage.metrics as metrics
from tqdm import tqdm
import time
# -----------------------------------------------------------------------------
import tensorflow as tf

import random
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from datetime import datetime
now = datetime.now()
print(f"Current time: {now.hour}:{now.minute}:{now.second}")

# ----------------------------- ESR-GAN Generator -----------------------------
def dense_block(x, filters=32, growth_rate=32):
    inputs = [x]
    for _ in range(4):  # 4 convs inside each dense block
        x = Conv2D(growth_rate, kernel_size=3, padding='same')(x)
        x = PReLU(shared_axes=[1, 2])(x)
        inputs.append(x)
        x = Concatenate()(inputs)
    return x

def RRDB_block(input_tensor):
    x = dense_block(input_tensor)
    x = dense_block(x)
    x = dense_block(x)
    #x = Conv2D(input_tensor.shape[-1], kernel_size=3, padding='same')(x)
    x = Conv2D(input_tensor.shape[-1], kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    return layers.add([input_tensor, x * 0.2])

#Generator model
def create_gen_ESRGAN(gen_ip, num_res_block):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)

    temp = layers
    for i in range(num_res_block):
        #layers = res_block(layers)
        layers = RRDB_block(layers)

    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5, fused=False)(layers)  # ***
    layers = add([layers,temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9,9), padding="same")(layers)
    return Model(inputs=gen_ip, outputs=op)



# -------------------------- SR-GAN Generator --------------------------------
#Define blocks to build the generator
def res_block(ip):
    res_model = Conv2D(64, (3,3), padding = "same")(ip)
    res_model = BatchNormalization(momentum = 0.5, fused=False)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)
    
    res_model = Conv2D(64, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization(momentum = 0.5, fused=False)(res_model)
    return add([ip,res_model])

def upscale_block(ip):
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D( size = 2, interpolation='bilinear')(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    return up_model

#Generator model
def create_gen(gen_ip, num_res_block):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)

    temp = layers
    for i in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5, fused=False)(layers)  # ***
    layers = add([layers,temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9,9), padding="same")(layers)
    return Model(inputs=gen_ip, outputs=op)



# -------------- Discriminator 
from keras.layers import Dropout
def discriminator_block_old(ip, filters, strides=1, bn=True, dropout_rate=0.25):
    disc_model = Conv2D(filters, (3,3), strides = strides, padding="same")(ip)
    if bn:
        disc_model = BatchNormalization( momentum=0.8, fused=False )(disc_model)
    disc_model = LeakyReLU( alpha=0.2 )(disc_model)
    disc_model = Dropout(dropout_rate)(disc_model)  # Add dropout to regularize
    return disc_model

def discriminator_block(ip, filters, strides=1, bn=True, dropout_rate=0.25, kernel_initializer='glorot_uniform'):
    disc_model = Conv2D(filters, (3,3), strides=strides, padding="same", kernel_initializer=kernel_initializer)(ip)
    if bn:
        disc_model = BatchNormalization(momentum=0.8, fused=False)(disc_model)
    disc_model = LeakyReLU(alpha=0.2)(disc_model)
    disc_model = Dropout(dropout_rate)(disc_model)
    return disc_model

#Descriminartor, as described in the original paper
def create_disc_SRGAN(disc_ip):
    df = 64
    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    return Model(disc_ip, validity)

def create_disc_ESRGAN(disc_ip):
    df = 128
    #d1 = discriminator_block(disc_ip, df, bn=False)
    d1 = discriminator_block(disc_ip, df, bn=False, kernel_initializer='he_normal')
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    
    d8_5 = Flatten()(d8)
    #d9 = Dense(df*16)(d8_5)
    d9 = Dense(df*16, kernel_initializer='he_normal')(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, kernel_initializer='he_normal')(d10)  # No sigmoid
    return Model(disc_ip, validity)


# VGG19 
def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet",include_top=False, input_shape=hr_shape)
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)
    
#Combined model
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    gen_features = vgg(gen_img)
    
    disc_model.trainable = False
    validity = disc_model(gen_img)
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])

def create_combined_with_pixel_loss(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    disc_model.trainable = False
    validity = disc_model(gen_img)
    vgg_features = vgg(gen_img)
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, vgg_features, gen_img])
# -----------------------------------------------------------------------------



# Load the images
lr_images = np.load(r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_entropy\Entrop_q1_sat_imgs_norm_64x64_4b.npy")
hr_images = np.load(r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\Data_entropy\Entrop_q1_drn_imgs_norm_256x256_4b.npy")
print("LR & HR shape", lr_images.shape, hr_images.shape)


lr_images = lr_images[:, :, :, :3] # lr_images[:904, :, :, :3] # lr_images[:511, :, :, :3] # 
hr_images = hr_images[:, :, :, :3] # hr_images[:904, :, :, :3] # hr_images[:511, :, :, :3] # 
print("LR & HR shape", lr_images.shape, hr_images.shape)

#@ >>>>>>>>>>>>>>>>>>>>>>>>>>
image_number = random.randint(0, len(lr_images)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(lr_images[image_number], (64, 64, 3))) # (250, 250, 3))) # (32, 32, 3)))
plt.subplot(122)
plt.imshow(np.reshape(hr_images[image_number], (256, 256, 3))) #(1000, 1000, 3))) # (128, 128, 3)))
plt.show() #'''


#Split to train and test
lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, test_size=0.20, random_state=42)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])
lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)


#generator = create_gen(lr_ip, num_res_block = 16)                             # *** SRGAN
generator = create_gen_ESRGAN(lr_ip, num_res_block = 16)                       # *** ESRGAN
#generator.summary()


#discriminator = create_disc_SRGAN(hr_ip)                                       # *** SRGAN
discriminator = create_disc_ESRGAN(hr_ip)                                      # *** ESRGAN

#opt_disc = Adam(learning_rate=2e-4, beta_1=0.9)
opt_disc = Adam(learning_rate=4e-4, beta_1=0.9, clipnorm=0.1)          ########   Tuning
#opt_disc = Adam(learning_rate=3e-4, beta_1=0.9, clipnorm=0.05)
#discriminator.compile(loss="binary_crossentropy", optimizer=opt_disc, metrics=["accuracy"])
### Fixing ESRGAN
from keras.losses import BinaryCrossentropy
bce = BinaryCrossentropy(from_logits=True)
discriminator.compile(loss=bce, optimizer=opt_disc, metrics=["accuracy"])
#discriminator.summary()

vgg = build_vgg((256,256,3))
#print(vgg.summary())
vgg.trainable = False

#opt_gen = Adam(learning_rate=1e-4, beta_1=0.5)
#opt_gen = Adam(learning_rate=5e-5, beta_1=0.5, clipnorm=1.0)  # Add gradient clipping
opt_gen = Adam(learning_rate=1e-4, beta_1=0.5, clipnorm=0.1)           ########    Tuning
#opt_gen = Adam(learning_rate=1e-4, beta_1=0.5, clipnorm=0.05)




gan_model = create_combined_with_pixel_loss(generator, discriminator, vgg, lr_ip, hr_ip)

gan_model.compile(
    loss=["binary_crossentropy", "mae", "mae"], # [adversarial, perceptual, pixel]
    #loss_weights=[1e-3, 1.0, 1.0],  # try 1.0 or 0.5 for perceptual vs pixel balance
    loss_weights=[1e-3, 0.5, 1.0],
    #loss_weights=[5e-4, 0.5, 1.0],
    #loss_weights=[5e-4, 0.5, 0.5],
    optimizer=opt_gen
)
gan_model.summary()


#Create a list of images for LR and HR in batches from which a batch of images
batch_size = 1  
train_lr_batches = []
train_hr_batches = []
for it in range(int(hr_train.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])


from keras import backend as K
def total_variation_loss_old_old(y_pred):
    return K.sum(K.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])) + \
           K.sum(K.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))

def total_variation_loss_old(y_pred):
    # Using TensorFlow operations directly
    tv_h = tf.reduce_sum(tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]))
    tv_w = tf.reduce_sum(tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))
    return tv_h + tv_w

def total_variation_loss(y_pred):
    tv_h = tf.reduce_sum(tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]) + 1e-8)
    tv_w = tf.reduce_sum(tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) + 1e-8)
    return tv_h + tv_w

# Initialize best loss to track
best_g_loss = float('inf') #float('inf')
lr_patience_counter = 0
current_lr = K.get_value(opt_gen.learning_rate)

# Training loop
start_time = time.perf_counter ()  # <---- TIME

history = {
    "g_loss": [],
    "d_loss": [],
    "test_psnr": [],
    "test_ssim": []
}
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='g_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='g_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)


epochs = 30
#Enumerate training over epochs
for e in range(epochs):
    # ------ Label smoothing
    real_label = np.random.uniform(0.8, 1.0, size=(batch_size, 1))
    fake_label = np.random.uniform(0.0, 0.2, size=(batch_size, 1))
    
    #Create empty lists to populate gen and disc losses. 
    g_losses = []
    d_losses = []
    
    #Enumerate training over batches. 
    for b in tqdm(range(len(train_hr_batches))):
        lr_imgs = train_lr_batches[b] #Fetch a batch of LR images for training
        hr_imgs = train_hr_batches[b] #Fetch a batch of HR images for training
        
        fake_imgs = generator.predict_on_batch(lr_imgs) #Fake images
        
        for _ in range(5):
            d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label + 1e-8)
            d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
        
        #Now, train the generator by fixing discriminator as non-trainable
        discriminator.trainable = False
        
        #Extract VGG features, to be used towards calculating loss
        image_features = vgg.predict(hr_imgs, verbose=0)
     
        #Train the generator via GAN. 
        gan_loss = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features, hr_imgs])
        # Extract the generator's total loss from the list of losses returned by train_on_batch
        
        g_loss = gan_loss[0]
        
        generated_imgs = generator.predict(lr_imgs, verbose=0)
        tv_loss_value = total_variation_loss(tf.constant(generated_imgs)).numpy() # Convert tensor to numpy array
        #g_loss = float(g_loss) + (1e-6 * float(tv_loss_value))  # Ensure scalar multiplication and addition
        
        #g_loss += 1e-4 * tv_loss_value
        g_loss += 1e-6 * tv_loss_value
        
        #Save losses to a list so we can average and report. 
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
    #Convert the list of losses to an array to make it easy to average    
    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)
    
    #Calculate the average losses for generator and discriminator
    g_loss = np.mean(g_losses)  # scalar
    d_loss = np.mean([d[0] if isinstance(d, (list, np.ndarray)) else d for d in d_losses])
    
    
    # Manual learning rate adjustment (mimicking ReduceLROnPlateau)
    if g_loss < best_g_loss:
        best_g_loss = g_loss
        lr_patience_counter = 0
    else:
        lr_patience_counter += 1
        
        if lr_patience_counter >= 5:
            new_lr = max(current_lr * 0.1, 1e-6)  # Factor = 0.1
            if new_lr < current_lr:
                K.set_value(opt_gen.learning_rate, new_lr)
                current_lr = new_lr
                print(f"Reduced learning rate to {new_lr} at epoch {e+1}")
                lr_patience_counter = 0
    
    
    # Evaluate PSNR on test set
    psnr_values = []
    ssim_values = []
    for i in range(len(hr_test)):
        hr_img = hr_test[i]
        gen_img = generator.predict(lr_test[i:i+1], verbose=0)[0]
        hr_img = np.clip(hr_img, 0, 1)
        gen_img = np.clip(gen_img, 0, 1)
        
        psnr_val = peak_signal_noise_ratio(hr_img, gen_img, data_range=1.0)
        psnr_values.append(psnr_val)
        
        # SSIM (on grayscale version)
        hr_gray = np.mean(hr_img, axis=2)
        gen_gray = np.mean(gen_img, axis=2)
        ssim_val = compare_ssim(hr_gray, gen_gray, data_range=1.0)
        ssim_values.append(ssim_val)
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    history["g_loss"].append(g_loss)
    history["d_loss"].append(d_loss)
    history["test_psnr"].append(avg_psnr)
    history["test_ssim"].append(avg_ssim)
    
    #Report the progress during training. 
    print(f"epoch: {e+1}  g_loss: {g_loss:.4f}  d_loss: {d_loss:.4f}  test_psnr: {avg_psnr:.2f} dB  test_ssim: {avg_ssim:.4f}")

    if (e+1) % 5 == 0: #Change the frequency for model saving, if needed
        #Save the generator after every n epochs (Usually 10 epochs)
        generator.save("gen_q4_e_"+ str(e+1) +".h5")

print("------> Running Time: ", time.perf_counter() - start_time, "sec  ", (time.perf_counter() - start_time)/60,"min")
# ------> Running Time:  12152.807356499994 sec   202.5467892916667 min
###############################################################################



# Plot loss curves
'''plt.figure(figsize=(12, 5))
plt.plot(history["g_loss"], label="Generator Loss")
plt.plot(history["d_loss"], label="Discriminator Loss")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show() #'''


fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(history["g_loss"], color='blue')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Generator Loss", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax2 = ax1.twinx()
ax2.plot(history["d_loss"], color='red')
ax2.set_ylabel("Discriminator Loss", color='k')
ax2.tick_params(axis='y', labelcolor='k')

plt.title("Training Loss per Epoch")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Plot PSNR
plt.figure(figsize=(8, 5))
plt.plot(history["test_psnr"], label="Test PSNR")
plt.title("Average Test PSNR per Epoch")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.grid(True)
plt.show()

# Plot SSIM
plt.figure(figsize=(8, 5))
plt.plot(history["test_ssim"], label="Test SSIM", color="purple")
plt.title("Average Test SSIM per Epoch")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.grid(True)
plt.legend()
plt.show()




###################################################################################
# Test - perform super resolution using either above or saved generator model
from keras.models import load_model
from numpy.random import randint

use_pretrained_generator = False

if use_pretrained_generator:
    #generator = load_model("D:/work/research_t/downscaling/code/HN_code/model_SRGAN_ndvi_q2_e50_replic/gen_q4_e_30.h5", compile=False)
    generator = load_model(r"D:/work/research_t/downscaling/code/HN_code/model_SRGAN_ndvi_q1_e30_replic/gen_q4_e_30.h5", compile=False)


[X1, X2] = [lr_test, hr_test]
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

# generate image from source
gen_image = generator.predict(src_image)


def stretch_image(img):
    # img: (H,W,3) normalized [0–1] or not
    out = np.zeros_like(img)
    for i in range(3):
        p2, p98 = np.percentile(img[:,:,i], (2, 98))
        out[:,:,i] = np.clip((img[:,:,i] - p2) / (p98 - p2 + 1e-6), 0, 1)
    return out

# plot all three images
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('LR Image')
plt.imshow(stretch_image(src_image[0,:,:,:])) #plt.imshow(src_image[0,:,:,:])
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(stretch_image(gen_image[0,:,:,:])) #plt.imshow(gen_image[0,:,:,:])
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(stretch_image(tar_image[0,:,:,:])) #plt.imshow(tar_image[0,:,:,:])
#plt.axis('off')
#plt.tight_layout()
plt.show()




mse = np.mean((tar_image - gen_image) ** 2)
psnr_hand = 100 if mse == 0 else 10 * np.log10(1.0 / (mse + 1e-10))  # Add epsilon to avoid division by zero
psnr_func = peak_signal_noise_ratio(tar_image, gen_image)
print("PSNR:", psnr_hand, psnr_func)

ssim1 = metrics.structural_similarity(tar_image[0], gen_image[0], win_size=3, channel_axis=-1, data_range=1.0)
ssim2 = metrics.structural_similarity(tar_image[0], gen_image[0], win_size=7, channel_axis=-1, data_range=1.0)
ssim3 = metrics.structural_similarity(tar_image[0], gen_image[0], win_size=75, channel_axis=-1, data_range=1.0)
print("SSIM (win_size=3): ", ssim1)
print("SSIM (win_size=7): ", ssim2)
print("SSIM (win_size=75): ", ssim3)

print(tar_image.shape)
print(gen_image.shape)





# %% 
###############################################################################
# Compute PSNR and SSIM for all test images
psnr_values, psnr1_values = [],[]
ssim_values, ssim1_values = [],[]
mae_values, mse_values = [],[]

for i in tqdm(range(len(hr_test))):
    # Extract high-res and generated images
    hr_img = hr_test[i]  # Ground truth
    gen_img = generator.predict(lr_test[i:i+1], verbose=0)[0]  # Predicted super-resolved image
    
    #gen_img = generator.predict(lr_test[i:i+1], verbose=0)[0]
    
    # Ensure images are in [0, 1] range
    hr_img = np.clip(hr_img, 0, 1) # Replace values < 0 with 0, and > 1 with 1.
    gen_img = np.clip(gen_img, 0, 1)
    
    # Compute PSNR by hand and function
    mae = np.mean(np.abs(hr_img - gen_img))
    mse = np.mean((hr_img - gen_img) ** 2)
    psnr = 100 if mse == 0 else 10 * np.log10(1.0 / (mse + 1e-10))  # Add epsilon to avoid division by zero
    psnr1 = peak_signal_noise_ratio(hr_img, gen_img)
    
    # Compute SSIM
    ssim = metrics.structural_similarity(hr_img, gen_img, win_size=3, channel_axis=-1, data_range=1.0)

    psnr_values.append(psnr)
    psnr1_values.append(psnr1)
    ssim_values.append(ssim)
    #ssim1_values.append(ssim1)
    mae_values.append(mae)
    mse_values.append(mse)

# Compute average scores
print(f"\nAverage Peak Signal-to-Noise Ratio (PSNR) (40-dB good): {np.mean(psnr_values):.2f} dB")
print(f"Average SSIM: {np.mean(ssim_values):.4f}")
# -----------------------------------------------------------------------------


print("Generator Loss:\n", history["g_loss"])
print("Discriminator Loss:\n", history["d_loss"])
print("Test PSNR:\n", history["test_psnr"])
print("Test SSIM:\n", history["test_ssim"])


print("psnr_values: \n", psnr_values)
print("ssim_values: \n", ssim_values)
print("mae_values: \n", mae_values)
print("mse_values: \n", mse_values)














































