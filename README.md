# Description

This repository contains the scripts and workflows associated with the manuscript “Assessing GAN Super-Resolution in Grasslands: The Role of Spatial Heterogeneity and Textural Complexity”.

The objective of this study is to evaluate how landscape heterogeneity influences the performance and robustness of image downscaling models in grassland ecosystems. Specifically, we assess how gradients in vegetation condition, spatial structure, and textural complexity affect super-resolution fidelity across three downscaling strategies: intra-sensor, cross-sensor generalization, and domain transfer. Model performance is compared across GAN-based architectures (SRGAN and ESRGAN) and a conventional interpolation baseline.

High-resolution imagery was collected using UAV flights over the Edwards Plateau ecoregion at the Carl and Bina Sue Martin – Texas A&M AgriLife Research Ranch, Texas, covering 1,553 ha of Mesquite-Oak savanna. UAV images were processed into 30 × 30 m tiles, classified based on vegetation health (NDVI), landscape structure (patch metrics), and texture (GLCM entropy), and paired with low-resolution PlanetScope imagery for model training and evaluation.

This repository includes scripts for:

- Preprocessing UAV and satellite imagery

- Tile extraction and classification

- Training and evaluating SRGAN and ESRGAN models

- Generating downscaled outputs and performance metrics (PSNR, SSIM)

The repository provides a complete landscape-aware pipeline for testing super-resolution performance under heterogeneous ecological conditions, supporting reproducibility and further research in remote sensing and ecological modeling.


## UAV Image Preprocessing and Raster Clipping

The Python code "pre_proc_v2b_auto2c.py" automates the preprocessing of UAV (drone) and satellite images, including:

- Loading shapefiles corresponding to UAV flight footprints and delineating flight boundaries.
- Generating an inner buffer along flight borders to avoid extracting image patches that include NoData or edge artifacts.
- Creating a fishnet grid of candidate points within the buffered flight polygons.
- Randomly selecting points while enforcing a minimum separation distance to reduce spatial autocorrelation.
- Generating square sampling polygons centered on the selected points.
- Clipping high-resolution (HR) raster imagery using the sampling polygons.
- Filtering extracted raster tiles based on the proportion of NoData values.
- Loading and clipping the corresponding low-resolution (LR) raster imagery using the same polygons.
- Ensuring that an equal number of paired HR and LR image tiles is obtained.

In a subsequent step, the extracted HR and LR image tiles are converted into NumPy (.npy) format using a separate Python script.

![image_alt](https://github.com/noayarae/Assessing-GAN-Super-Resolution-in-Grasslands/blob/main/imgs/preprocess_flowchart.png?raw=true)

### Requirements

- Python 3.x
- geopandas
- shapely
- rasterio
- fiona
- numpy
- tqdm
- scipy

### Example Usage

Set the drone flight index:

```python
dfg = 16  # 0-based index for flight







