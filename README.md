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

This Python project automates the preprocessing of UAV (drone) and satellite images, including:

- Loading shapefiles and generating buffer zones.
- Creating a fishnet grid of points within the buffered polygons.
- Selecting random points with minimum distance constraints.
- Generating square polygons around points.
- Clipping high-resolution (HR) raster images using polygons.
- Filtering raster tiles based on NoData values.

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







