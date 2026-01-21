"""
Created on Wed Jul 16 17:11:05 2025
@author: efrain.noa-yarasca
environment: tf
"""

import os
import numpy as np
import pandas as pd
from skimage import img_as_float
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm


###############################################################################
def compute_vegetation_indices(red, nir):
    # Ensure inputs are float for calculations
    red, nir = map(img_as_float, [red, nir])
    # Avoid division by zero or near-zero values
    sum_red_nir = red + nir
    ndvi = np.where(sum_red_nir == 0, 0, (nir - red) / (sum_red_nir)) # Handle division by zero
    savi = np.where((red + nir + 0.5) == 0, 0, ((nir - red) / (red + nir + 0.5)) * 1.5) # Handle division by zero
    # Assuming Blue band is 0 if not available, adjust if your data has a Blue band
    evi_denominator = nir + 6 * red - 7.5 * 0 + 1 # Blue = 0 (not used here)
    evi = np.where(evi_denominator == 0, 0, 2.5 * (nir - red) / (evi_denominator)) # Handle division by zero

    # Calculate means, ignoring NaNs that might result from division by zero
    return np.nanmean(ndvi), np.nanmean(savi), np.nanmean(evi)


def compute_morphology_old(ndvi, ndvi_threshold=0.5, min_size=20):
    # Ensure ndvi is a numpy array before comparison
    ndvi_array = np.asarray(ndvi)
    binary = ndvi_array > ndvi_threshold
    binary = remove_small_objects(binary, min_size=min_size)
    label_img = label(binary)
    props = regionprops(label_img)
    patch_count = len(props)
    mean_size = np.mean([p.area for p in props]) if props else 0
    #return patch_count, mean_object_size, binary
    return patch_count, mean_size, binary

def compute_morphology(ndvi, ndvi_threshold=0.5, min_size=200):
    # Ensure ndvi is a numpy array before comparison
    ndvi_array = np.asarray(ndvi)
    binary = ndvi_array > ndvi_threshold
    binary = remove_small_objects(binary, min_size=min_size)
    label_img = label(binary)
    props = regionprops(label_img)
    patch_count = len(props)
    mean_size = np.mean([p.area for p in props]) if props else 0

    # Calculate additional morphology metrics
    patch_areas = [p.area for p in props]
    std_dev_size = np.std(patch_areas) if patch_areas else 0 # Standard deviation of patch sizes

    patch_eccentricities = [p.eccentricity for p in props]
    mean_eccentricity = np.mean(patch_eccentricities) if patch_eccentricities else 0 # Mean eccentricity
    std_dev_eccentricity = np.std(patch_eccentricities) if patch_eccentricities else 0 # Standard deviation of eccentricity

    # Calculate standard deviation of NDVI within patches
    ndvi_in_patches = ndvi_array[binary] # Select NDVI values where binary mask is True
    std_dev_ndvi_in_patches = np.std(ndvi_in_patches) if ndvi_in_patches.size > 0 else 0 # Standard deviation

    #return patch_count, mean_size, std_dev_size, mean_eccentricity, std_dev_eccentricity, std_dev_ndvi_in_patches, binary
    return patch_count, mean_size, std_dev_size, std_dev_ndvi_in_patches, binary

def compute_texture_features(image, distances=[1, 33], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    image = img_as_ubyte(image)
    texture = {}

    for d in distances:
        glcm = graycomatrix(image, distances=[d], angles=angles, symmetric=True, normed=True)

        # Entropy
        with np.errstate(divide='ignore', invalid='ignore'):
            glcm_log = np.log2(glcm + 1e-10)
            entropy = -np.sum(glcm * glcm_log, axis=(0, 1))
        texture[f'Entropy_d{d}'] = np.mean(entropy)

        # Other GLCM properties
        for prop in ['contrast', 'homogeneity', 'correlation']:
            val = graycoprops(glcm, prop).mean()
            texture[f'{prop.capitalize()}_d{d}'] = val

    return texture


# folder with your tiles
input_folder = r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\hr_imgs" # **UPDATE THIS PATH**
results = []

# Check if the input folder exists
# rando mly pick a few tiles to visualize
#tile_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]
#visualize_tiles = set(np.random.choice(tile_files, size=min(5, len(tile_files)), replace=False))
#visualize_tiles = {'img_00001_polygon_0.tif', 'img_00002_polygon_1.tif', 'img_00003_polygon_3.tif'}
#visualize_tiles

#list_tif_files = os.listdir(input_folder)
list_tif_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".tif")]
print(len(list_tif_files))

# Save intermediate images if you want to plot later
#intermediate_data = {}  # store {fname: (rgb_img, ndvi, binary_mask)}


# --- STEP 1
# ========================= Compute all tile features =========================
# Compute: 'NDVI_mean', 'SAVI_mean', 'EVI_mean', 'Tree_patch_count', 'Mean_object_size', 'Std_dev_patch_size', 
#          'Std_dev_ndvi_in_patches', 'Entropy_d1', 'Contrast_d1', 'Homogeneity_d1', 'Correlation_d1',
#          'Entropy_d33', 'Contrast_d33', 'Homogeneity_d33', 'Correlation_d33'

#for fname in tqdm(list_tif_files[:200]): # <------------ SET Number-of-Tiles
for fname in tqdm(list_tif_files): # <------------ SET Number-of-Tiles
    path = os.path.join(input_folder, fname)
    
    with rasterio.open(path) as src:
        try:
            # Attempt to read bands for RGB and NIR
            # Assuming band order is Red, Green, Blue, NIR (1, 2, 3, 4)
            red = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32)
            blue = src.read(3).astype(np.float32)
            nir = src.read(4).astype(np.float32)

            # Create an RGB image for visualization (normalize to 0-1 for plotting)
            # Stack bands and normalize by max value or known data range if available
            # Simple normalization by max value in each band for display purposes
            rgb_img = np.dstack((red, green, blue))
            # Normalize for display - adjust based on your data's actual value range
            rgb_img = rgb_img / np.max(rgb_img) if np.max(rgb_img) > 0 else rgb_img

        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

    ndvi_mean, savi_mean, evi_mean = compute_vegetation_indices(red, nir)
    #print("ndvi_mean, savi_mean, evi_mean: ", ndvi_mean, savi_mean, evi_mean)  # To test

    # Recalculate ndvi for morphology function
    ndvi = (nir - red) / (nir + red + 1e-10)
    #patch_count, mean_size, SD_size, mean_eccentricity, SD_eccentricity, SD_ndvi_in_patches, binary_mask = compute_morphology(ndvi)
    patch_count, mean_size, SD_size, SD_ndvi_in_patches, binary_mask = compute_morphology(ndvi)
    # Texture
    texture_feats = compute_texture_features(ndvi, distances=[1, 33]) # --->  Compute Texture
        
    result = {
        'Tile_ID': fname,
        # ----- Vegetation health
        'NDVI_mean': ndvi_mean,
        'SAVI_mean': savi_mean,
        'EVI_mean': evi_mean,
        # ----- Landscape structure
        'Tree_patch_count': patch_count,
        'Mean_object_size': mean_size,
        'Std_dev_patch_size': SD_size,
        #'Mean_eccentricity': mean_eccentricity,
        #'Std_dev_eccentricity': SD_eccentricity,
        'Std_dev_ndvi_in_patches': SD_ndvi_in_patches,
        # Class (pending to implement)
        #'Class': None  # optional
    }
    result.update(texture_feats)
    results.append(result)

df = pd.DataFrame(results)   # Convert as df

# save temporally the results of tiles and all features
output_df = r"D:\work\research_t\downscaling\Results\000_tiles_w_all_features.csv"
#df.to_csv(output_df, index=False)
########################### End: compute tile features ########################





#df = df.drop(columns=["Cluster"])
df_csv_features = df.copy()





# --- STEP 2
###############################################################################
# Read the csv file if you already computed the features. 
# Because the previous step runs in about 25 minutes
df_csv_features = pd.read_csv(r"D:\work\research_t\downscaling\Results\000_tiles_w_all_features.csv")
#df_csv_features = pd.read_csv(r"D:\work\research_t\downscaling\Results\000_tiles_w_all_features_verified.csv")



# Cluster Analysis (Elbow method)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def plot_elbow(df1, features, title, k_max=10):
    # Standardize
    X_scaled = StandardScaler().fit_transform(df1[features])

    # Compute inertia
    k_range = range(1, k_max + 1)
    inertia = [
        KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
        for k in k_range
    ]

    # Plot
    plt.figure(figsize=(6, 4.5), dpi=300)
    plt.plot(k_range, inertia, linestyle='--', linewidth=0.9, marker='s',
             markersize=7, markerfacecolor='#b3b3ff', markeredgecolor='b')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title(title)
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.6)
    plt.show()

# 1. Structural features
structure_features = ['Tree_patch_count', 'Mean_object_size', 'Std_dev_patch_size']
title_st = 'Elbow Method for Clustering Structurally \n Characterized Tiles'
plot_elbow(df_csv_features, structure_features, title=title_st)

# 2. Texture features
texture_features = ['Entropy_d33', 'Contrast_d33', 'Homogeneity_d33', 'Correlation_d33']
title_text = 'Elbow Method for Clustering Texture-\nCharacterized Tile Images'
plot_elbow(df_csv_features, texture_features, title=title_text)
#'''
###############################################################################






# --- STEP 3
###############################################################################
# Clustering using target columns & plot using PCA axes
# k = 3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_kmeans_clustering(df, features, cluster_col, cluster_names, title, k=3, xlim=None):
    # --- Standardize ---
    X_scaled = StandardScaler().fit_transform(df[features])
    # --- KMeans ---
    kmeans = KMeans(n_clusters=k, random_state=42)
    df[cluster_col] = kmeans.fit_predict(X_scaled)
    # --- PCA Projection ---
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    # Plot settings
    colors      = ['#7fc97f', 'lightblue', '#fdc086']
    edge_colors = ['#006600', 'blue', 'red']
    markers     = ['o', 'o', 'o']

    # --- Plot ---
    plt.figure(figsize=(6, 5), dpi=400)
    for i in range(k):
        idx = df[cluster_col] == i
        plt.scatter(
            X_pca[idx, 0], X_pca[idx, 1],
            label=cluster_names[i],
            c=colors[i],
            marker=markers[i],
            s=50,
            alpha=0.8,
            edgecolors=edge_colors[i],
            linewidths=0.9
        )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    plt.show()

    # --- Print cluster counts ---
    print("\nRows per cluster:")
    print(df[cluster_col].value_counts().sort_index())


# 1. Structural
cluster_col = 'cluster_st'
cluster_names = [
    'Sparse & Highly Fragmented Woody Cover',
    'Dense & Interspersed Woody Mosaic',
    'Large & Consolidated Woody Patches'
]
structure_features = ['Tree_patch_count', 'Mean_object_size', 'Std_dev_patch_size']
title_st = "Structural Feature Clustering (PCA Projection)"
run_kmeans_clustering(df_csv_features, structure_features,cluster_col, cluster_names,title=title_st,k=3, xlim=(-2.5, 14))

# 2. Texture
cluster_col = 'cluster_tx'
cluster_names = [
    'Texture C1',
    'Texture C2',
    'Texture C3'
    ]
texture_features = ['Entropy_d33', 'Contrast_d33', 'Homogeneity_d33', 'Correlation_d33']
title_text = "Texture Feature Clustering (PCA Projection)"
run_kmeans_clustering(df_csv_features, texture_features,cluster_col, cluster_names,title=title_text,k=3, xlim=None)
# ---------------------------- End: Cluster k=3 ------------------------------




# --- STEP 4
###############################################################################
# ---------------  Plot RGB, NDVI, and Binary Mask for specific Cluster -------------
# For Figure 6. Landscape structure–derived clusters: (a) Cluster 1: Sparse & Fragmented Woody Patches, (b) Cluster 2: Dispersed Woody Mosaics, and (c) Cluster 3: Dense & Clumped Woody Dominance.

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
import random
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

def compute_ndvi(red, nir):
    return (nir - red) / (nir + red + 1e-10)

def compute_binary_mask(ndvi, ndvi_threshold=0.5, min_size=200):
    ndvi_array = np.asarray(ndvi)
    binary = ndvi_array > ndvi_threshold
    binary = remove_small_objects(binary, min_size=min_size)
    return binary

def plot_one_tile(df1, input_folder, cluster_col, cluster=None, tile_name=None, subplot_label=None):
    """
    Plots a tile showing RGB, NDVI, and Binary Mask.
            If tile_name is given, it uses that image directly.
            If tile_name is None, it randomly selects a tile from the given cluster.
    """
    if tile_name:
        # Specific tile
        if tile_name not in df1['Tile_ID'].values:
            print(f"Tile {tile_name} not found in dataframe.")
            return
        row = df1[df1['Tile_ID'] == tile_name].iloc[0]
        cluster = row[cluster_col]
    else:
        # Random tile from cluster
        if cluster is None:
            print("Please provide a cluster or a tile_name.")
            return
        tiles_in_cluster = df1[df1[cluster_col] == cluster]
        if tiles_in_cluster.empty:
            print(f"No tiles found for cluster {cluster}")
            return
        row = tiles_in_cluster.sample(1).iloc[0]                               # Random selection is here
        print(row)
        tile_name = row['Tile_ID']
    path = os.path.join(input_folder, tile_name)

    with rasterio.open(path) as src:
        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)
    rgb_img = np.dstack((red, green, blue))
    rgb_img /= np.max(rgb_img) if np.max(rgb_img) > 0 else 1
    ndvi = compute_ndvi(red, nir)
    binary_mask = compute_binary_mask(ndvi)


    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300) #, constrained_layout=True)
    plt.subplots_adjust(wspace=0.15)
    fig.suptitle(f"Tile: {tile_name} (cluster_col {cluster})", fontsize=14, y=1.02)

    # RGB
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"RGB Image {tile_name}", fontsize=10)
    axes[0].text(0.5, -0.02, subplot_label, transform=axes[0].transAxes,fontsize=12, ha='center', va='top')
    axes[0].axis('off')

    # NDVI
    ndvi_clipped = np.clip(ndvi, -1, 1)
    ndvi_plot = axes[1].imshow(ndvi_clipped, cmap='RdYlGn')
    axes[1].set_title(f"NDVI {tile_name}", fontsize=10)
    axes[1].axis('off')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(ndvi_plot, cax=cax)

    # Binary Mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title(f"Binary Mask (NDVI>0.5) {tile_name}", fontsize=10)
    axes[2].axis('off')
    plt.show()


# For structure
clust_col = 'cluster_st'
plot_one_tile(df_csv_features, input_folder, clust_col, cluster=0, subplot_label='(a)')
plot_one_tile(df_csv_features, input_folder, clust_col, cluster=1, subplot_label='(b)')
plot_one_tile(df_csv_features, input_folder, clust_col, cluster=2, subplot_label='(c)')

# plot especific tiles
plot_one_tile(df_csv_features, input_folder, clust_col, tile_name="img_00930_polygon_217.tif")
plot_one_tile(df_csv_features, input_folder, clust_col, tile_name="img_02151_polygon_206.tif")
# -----------------------------------------------------------------------------





# --- STEP 5
###############################################################################
### Get Statistics of cluster-based features
# For Table 2. Statistics and landscape features of image tiles grouped by structure-based clustering using patch-level metrics: tree patch area, number of patches, and patch area variability
#     Table 3.
def summarize_clusters(df, cluster_col, columns):
    # Count rows per cluster
    print(f"\nNumber of rows per cluster ({cluster_col}):")
    print(df[cluster_col].value_counts().sort_index())

    # Compute mean and std
    grouped = df.groupby(cluster_col)[columns].agg(['mean', 'std'])

    for cluster, data in grouped.iterrows():
        print(f"\nCluster {cluster}:")
        for col in columns:
            mean, std = data[(col, 'mean')], data[(col, 'std')]
            print(f"  {col:>25}: mean = {mean:.3f}, std = {std:.3f}")


# --- STRUCTURE ---
structure_cols = ['Tree_patch_count','Mean_object_size','Std_dev_patch_size']
summarize_clusters(df_csv_features, 'cluster_st', structure_cols)

# --- TEXTURE ---
texture_cols = ['Entropy_d33', 'Contrast_d33', 'Homogeneity_d33', 'Correlation_d33']
summarize_clusters(df_csv_features, 'cluster_tx', texture_cols)
# -----------------------------------------------------------------------------






# --- STEP 6
# We load all input images and use the CSV files (containing the extracted features and their assigned cluster labels)
# to select (separate) the images according to the clusters and strategy (structural or textural).
# At the end we saved the segmented images as npy arrays
###############################################################################
# ----------------- Read tile images for especific cluster --------------------
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import rasterio
from tqdm import tqdm

# Folder paths
hr_folder = r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\hr_imgs"
lr_folder = r"D:\work\research_t\downscaling\put_all_imgs\Tiles_30m_set2\lr_imgs"

# Select tiles belonging to cluster 0
tile_ids_cluster0 = df_csv_features[df_csv_features['cluster_st'] == 0]['Tile_ID'].tolist()   # <-------  SET Cluster

# Containers
hr_images = []
lr_images = []
tile_rows_found = [] # tile_ids_found = []

# Read HR and LR images
print(f"Reading {len(tile_ids_cluster0)} tiles...")
#for tile_id in tqdm(tile_ids_cluster0):
for idx, tile_id in enumerate(tqdm(tile_ids_cluster0)):
    if idx % 200 == 0:
        print(f"[{idx}] Processing Tile_ID: {tile_id}")
        
    hr_path = os.path.join(hr_folder, tile_id)
    lr_path = os.path.join(lr_folder, tile_id)
    if os.path.exists(hr_path) and os.path.exists(lr_path):
        with rasterio.open(hr_path) as src:
            hr_img = src.read([1,2,3,4]).transpose(1,2,0).astype(np.float32)
        with rasterio.open(lr_path) as src:
            lr_img = src.read([1,2,3,4]).transpose(1,2,0).astype(np.float32)
        hr_images.append(hr_img)
        lr_images.append(lr_img)
        
        # Save the full row of df for this tile
        #tile_ids_found.append(tile_id)
        row = df_csv_features[df_csv_features['Tile_ID'] == tile_id].iloc[0]
        tile_rows_found.append(row)

hr_images = np.array(hr_images)
lr_images = np.array(lr_images)
print(f"Loaded {len(hr_images)} valid HR/LR image pairs.")

# Save & Store dataframe
df_tiles_found = pd.DataFrame(tile_rows_found)
#csv_file = "D:/work/research_t/downscaling/put_all_imgs/Tiles_30m_set2/data_texture/df_cluster_2.csv"
#df_tiles_found.to_csv(csv_file, index=False)   #                 <--------- SET cluster-file to save



# ----------- Normalization
def normalize_images(images, name=""):
    min_vals = images.min(axis=(0,1,2))
    max_vals = images.max(axis=(0,1,2))
    print(f"{name} per-channel min:", min_vals)
    print(f"{name} per-channel max:", max_vals)
    images_norm = (images - min_vals) / (max_vals - min_vals + 1e-6)
    print(f"{name} Norm per-channel min:", images_norm.min(axis=(0,1,2)))
    print(f"{name} Norm per-channel max:", images_norm.max(axis=(0,1,2)))    
    return images_norm

lr_images_norm = normalize_images(lr_images, "LR")
hr_images_norm = normalize_images(hr_images, "HR")
print(lr_images_norm.shape)
print(hr_images_norm.shape)



# ----------- Resizing
import cv2
target_size = (64, 64)
# OpenCV expects (width, height), so (250,250)
hr_images_64x64 = np.zeros((hr_images_norm.shape[0], target_size[1], target_size[0], hr_images_norm.shape[-1]), dtype=hr_images_norm.dtype)
for i in tqdm(range(hr_images_norm.shape[0]), desc="Resizing HR images with cv2"):
    for c in range(hr_images_norm.shape[-1]):
        hr_images_64x64[i, :, :, c] = cv2.resize(
            hr_images_norm[i, :, :, c],
            target_size,
            interpolation=cv2.INTER_CUBIC
        )
print("Resized HR shape:", hr_images_64x64.shape)

target_size = (256, 256)
# OpenCV expects (width, height), so (250,250)
hr_images_256x256 = np.zeros((hr_images_norm.shape[0], target_size[1], target_size[0], hr_images_norm.shape[-1]), dtype=hr_images_norm.dtype)
for i in tqdm(range(hr_images_norm.shape[0]), desc="Resizing HR images with cv2"):
    for c in range(hr_images_norm.shape[-1]):
        hr_images_256x256[i, :, :, c] = cv2.resize(
            hr_images_norm[i, :, :, c],
            target_size,
            interpolation=cv2.INTER_CUBIC
        )
print("Resized HR shape:", hr_images_256x256.shape)

target_size = (64, 64)
# OpenCV expects (width, height), so (250,250)
lr_images_64x64 = np.zeros((lr_images_norm.shape[0], target_size[1], target_size[0], lr_images_norm.shape[-1]), dtype=lr_images_norm.dtype)
for i in tqdm(range(lr_images_norm.shape[0]), desc="Resizing LR images with cv2"):
    for c in range(lr_images_norm.shape[-1]):
        lr_images_64x64[i, :, :, c] = cv2.resize(
            lr_images_norm[i, :, :, c],
            target_size,
            interpolation=cv2.INTER_CUBIC
        )
print("Resized HR shape:", lr_images_64x64.shape)

# ---------------- Save npy files
cluster = 0                                                 # <-------------  SET Cluster 
np.save(f'D:/work/research_t/downscaling/put_all_imgs/Tiles_30m_set2/data_texture/Tex_C{cluster}_sat_imgs_norm_10x10_4b.npy', lr_images_norm)
np.save(f'D:/work/research_t/downscaling/put_all_imgs/Tiles_30m_set2/data_texture/Tex_C{cluster}_sat_imgs_norm_64x64_4b.npy', lr_images_64x64)
np.save(f'D:/work/research_t/downscaling/put_all_imgs/Tiles_30m_set2/data_texture/Tex_C{cluster}_drn_imgs_norm_64x64_4b.npy', hr_images_64x64)
np.save(f'D:/work/research_t/downscaling/put_all_imgs/Tiles_30m_set2/data_texture/Tex_C{cluster}_drn_imgs_norm_256x256_4b.npy', hr_images_256x256)
np.save(f'D:/work/research_t/downscaling/put_all_imgs/Tiles_30m_set2/data_texture/Tex_C{cluster}_drn_imgs_norm_1000x1000_4b.npy', hr_images_norm)






# === STEP 7: Visualize (Real color) random 3 sample  ===
# === STEP 7.1: Visualize (Real color) random sample AFTER normalization ===
def stretch_image(img):
    # img: (H,W,3) normalized [0–1] or not
    out = np.zeros_like(img)
    for i in range(3):
        p2, p98 = np.percentile(img[:,:,i], (2, 98))
        out[:,:,i] = np.clip((img[:,:,i] - p2) / (p98 - p2 + 1e-6), 0, 1)
    return out

image_number = random.randint(0, len(lr_images_norm)-1) # 318 # 
tile_id = df_tiles_found.iloc[image_number]['Tile_ID']
lr_rgb = stretch_image(lr_images_norm[image_number][:,:,0:3])
lr_rgb_64x64 = stretch_image(lr_images_64x64[image_number][:,:,0:3])
hr_rgb_64x64 = stretch_image(hr_images_64x64[image_number][:,:,0:3])
hr_rgb_256x256 = stretch_image(hr_images_256x256[image_number][:,:,0:3])
hr_rgb = stretch_image(hr_images_norm[image_number][:,:,0:3])


plt.figure(figsize=(25,5)) # (18,6))
plt.subplot(1,5,1)
plt.imshow(lr_rgb)
plt.title(f'LR Img (RGB, Stretch)-C{cluster} \n {tile_id}',fontsize=16)
plt.subplot(1,5,2)
plt.imshow(lr_rgb_64x64)
plt.title(f'LR Resized Img (RGB, Stretch)-C{cluster} \n {tile_id}',fontsize=16)
plt.subplot(1,5,3)
plt.imshow(hr_rgb_64x64)
plt.title(f'HR Resized Img (RGB, Stretch)-C{cluster} \n {tile_id}',fontsize=16)
plt.subplot(1,5,4)
plt.imshow(hr_rgb_256x256)
plt.title(f'HR Resized Img (RGB, Stretch)-C{cluster} \n {tile_id}',fontsize=16)
plt.subplot(1,5,5)
plt.imshow(hr_rgb)
plt.title(f'HR Img (RGB, Stretch)-C{cluster} \n {tile_id}',fontsize=16)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------

























































