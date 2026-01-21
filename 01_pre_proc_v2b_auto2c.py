"""
Environment:  tf_clone
"""
# Import libraries 
import os, shutil
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
import rasterio
from rasterio.mask import mask
import fiona
#pip install fiona 
from rasterio.warp import reproject, Resampling
import time
from tqdm import tqdm

def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"
###############################################################################




### -------------------- Set file names ---------------------------------------
### List of shp files of flight borders (56 flighs. See: D:\work\research_t\downscaling\[compute_time.xlsx]Flight_dates)
border_files = ['border_01_E1.shp', 'border_02_N.shp', 'border_03_NE.shp', 'border_04_S1.shp', 'border_05_S2b.shp',
                'border_06_E2.shp', 'border_07_W1.shp', 'border_08_W2.shp', 'border_09_E.shp', 'border_10_N.shp',
                'border_11_W.shp', 'border_12_S.shp', 'border_13_SE.shp', 'border_14_SW1.shp', 'border_15_C.shp',
                'border_16_N.shp', 'border_17_NE.shp', 'border_18_C.shp', 'border_19_NW.shp', 'border_20_SW.shp',
                'border_21_NE.shp', 'border_22_S.shp', 'border_23_SE.shp', 'border_24_SE_L1.shp', 'border_25_SE_L2.shp',
                'border_26_C.shp', 'border_27_N.shp', 'border_28_NE.shp', 'border_29_S.shp', 'border_30_SW.shp',
                'border_31_W.shp', 'border_32_Lop1_S.shp', 'border_33_Lop2_E.shp', 'border_34_Lop3_N.shp', 'border_35_Lop4_C.shp',
                'border_36_Prugel_NW.shp', 'border_37_Prugel1_Lin.shp', 'border_38_Lop3_N.shp', 'border_39_Lop4_C.shp', 'border_40_Lop1__2r4_1.shp',
                'border_41_Lop1_2r4_2.shp', 'border_42_Lop1.shp', 'border_43_Lop2_2r_1.shp', 'border_44_Lop2_2r_2.shp', 'border_45_NW.shp', 
                'border_46_Prugel1.shp', 'border_47_Lop_C.shp', 'border_48_Lop_S.shp', 'border_49_Lop_E.shp', 'border_50_Lop_Corr.shp', 
                'border_51_N_SEC.shp', 'border_52_N_SSE.shp', 'border_53_N_SWC.shp', 'border_54_N_Corr_E.shp', 'border_55_N_Corr_N.shp', 
                'border_56_N_Corr_W.shp']

drone_tif_files = ['01_E1_r4.tif', '02_N_r.tif', '03_NE_r.tif', '04_S1_r.tif', '05_S2_r.tif', 
                   '06_E2_r.tif', '07_W1_r.tif', '08_W2_r.tif', '09_E_r.tif', '2023_06_13_10_N_r.tif', 
                   '2023_06_13_11_W_r.tif', '2023_07_26_12_S_r.tif', '2023_07_26_13_SE_r.tif', '2023_07_26_14_SW_r.tif', '2023_07_27_15_C_r.tif', 
                   '2023_07_27_16_N_r.tif', '2023_07_27_17_NE_r.tif', '2023_11_14_18_C_r.tif', '2023_11_14_19_NW_r.tif', '2023_11_14_20_SW_r.tif', 
                   '2023_11_15_21_NE_r.tif', '2023_11_15_22_S_r.tif', '2023_11_15_23_SE_r.tif', '2023_11_17_24_SE_L1_r.tif', '2023_11_17_25_SE_L2_r.tif', 
                   
                   '2023_12_05_26_C_r.tif', '2023_12_05_27_N_r.tif', '2023_12_05_28_NE_r.tif', '2023_12_06_29_S_r.tif', '2023_12_06_30_SW_r.tif', 
                   '2023_12_06_31_W_r.tif', '2024_01_09_32_Lop1_S_r.tif', '2024_01_09_33_Lop2_E_r.tif', '2024_01_09_34_Lop3_N_r.tif', '2024_01_09_35_Lop4_C_r.tif', 
                   '2024_01_10_36_Prugel_NW_r.tif', '2024_01_10_37_Prugel1_Lin_r.tif', '2024_01_10_38_Lop3_N_r.tif', '2024_01_10_39_Lop4_C_r.tif', '2024_01_18_40_Lop1_2r4_1_r.tif', 
                   '2024_01_18_41_Lop1_2r4_2_r.tif', '2024_01_18_42_Lop1_r.tif', '2024_01_18_43_Lop2_2r_1_r.tif', '2024_01_18_44_Lop2_2r_2_r.tif', '2024_01_19_45_NW_r.tif', 
                   '2024_01_19_46_Prugel1_r.tif', '2024_05_18_47_Lop_C_r.tif', '2024_05_18_48_Lop_S_r.tif', '2024_05_19_49_Lop_E_r.tif', '2024_05_19_50_Lop_Corr_r.tif',
                   
                   '2024_06_13_51_N_SEC_r.tif', '2024_06_13_52_N_SSE_r.tif', '2024_06_13_53_N_SWC_r.tif', '2024_06_14_54_N_Corr_E_r.tif', '2024_06_14_55_N_Corr_N_r.tif', 
                   '2024_06_14_56_N_Corr_W_r.tif']

List_LR_sub_folder = ['01_2023-04-03_psscene_analytic_sr_udm2', '02_2023-06-09_psscene_analytic_sr_udm2', '03_2023-07-20_psscene_analytic_sr_udm2',
                   '04_2023-11-15_psscene_analytic_sr_udm2', '05_2023-12-05_psscene_analytic_sr_udm2', '06_2024-01-10_psscene_analytic_sr_udm2',
                   '07_2024-01-20_psscene_analytic_sr_udm2', '08_2024-05-19_psscene_analytic_sr_udm2', '09_2024-06-06_psscene_analytic_sr_udm2']
# This is to allocate UAV images to corresponding Planet, because planet are not in same frequency than UAV
number_of_times_to_match_LR_to_HR = [8, 3, 6, 8, 6, 8, 7, 4, 6]
ext_List_LR_sub_folder = [item for item, count in zip(List_LR_sub_folder, number_of_times_to_match_LR_to_HR) for _ in range(count)]
print(len(ext_List_LR_sub_folder))
# ---


### Inputs (old fashion)
if_01_border_file = "D:/work/research_t/downscaling/ForEfrain/borders/border_05_S2b.shp"
if_02_HR_raster_img = "D:/work/research_t/downscaling/ForEfrain/wf_p_resampled/05_S2_r.tif"
if_03_LR_raster_img = "D:/work/research_t/downscaling/ForEfrain/Planet_select/01_2023-04-03_psscene_analytic_sr_udm2/composite.tif"

# FOLDER inputs
it_01_square_pol_folder = "D:/work/research_t/downscaling/pre_proc_imgs/flight_05/f05_polyg" # For polygons
it_02_HR_tif_folder = "D:/work/research_t/downscaling/pre_proc_imgs/flight_05/f05_hr_tif/"   # For HR imgs
it_03_LR_tif_folder = "D:/work/research_t/downscaling/pre_proc_imgs/flight_05/f05_lr_tif/"   # For LR imgs

### Outputs
of_01_buffer = "D:/work/research_t/downscaling/pre_proc_imgs/flight_05/f05_buff.shp"
of_02_rdm_pts = "D:/work/research_t/downscaling/pre_proc_imgs/flight_05/f05_rdm_pts.shp"
of_03_rdm_sqs = "D:/work/research_t/downscaling/pre_proc_imgs/flight_05/f05_rdm_sqs.shp"
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
### more automatized, providing just the index
dfg = 16 #55 #18     #  <-----  SET the Dron-Flight tif number (Set 0 For flight 1)

### SET input Files: Border.shp, UAV-file, and Planet-file
if_01_border_file = "D:/work/research_t/downscaling/ForEfrain/borders/" + border_files[dfg]
if_02_HR_raster_img = "D:/work/research_t/downscaling/ForEfrain/wf_p_resampled_3x3cm/" + drone_tif_files[dfg]
#if_02_HR_raster_img = "D:/work/research_t/downscaling/ForEfrain/wf_p_resampled_2/" + drone_tif_files[dfg]
if_03_LR_raster_img = "D:/work/research_t/downscaling/ForEfrain/Planet_select/" + ext_List_LR_sub_folder[dfg] +"/composite.tif"



### Create folder for outputs

#father_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs/flight_0"+ str(dfg+1)
#father_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_2/flight_0"+ str(dfg+1)
#father_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cmb/flight_0"+ str(dfg+1)
#father_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cm_hn/flight_0"+ str(dfg+1)
father_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cm_30m/flight_0"+ str(dfg+1)
#os.makedirs(father_folder_path, exist_ok=True)
shutil.rmtree(father_folder_path) if os.path.exists(father_folder_path) else None; os.makedirs(father_folder_path)

#child1_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs/flight_0"+ str(dfg+1)+ "/f0"+str(dfg+1)+"_polyg"
#child1_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_2/flight_0"+ str(dfg+1)+ "/f0"+str(dfg+1)+"_polyg"
#child1_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cmb/flight_0"+ str(dfg+1)+ "/f0"+str(dfg+1)+"_polyg"
#child1_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cm_hn/flight_0"+ str(dfg+1)+ "/f0"+str(dfg+1)+"_polyg"
child1_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cm_30m/flight_0"+ str(dfg+1)+ "/f0"+str(dfg+1)+"_polyg"
os.makedirs(child1_folder_path, exist_ok=True)
#child2_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs/flight_0"+ str(dfg+1)+"/f0"+str(dfg+1)+"_hr_tif/"
#child2_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_2/flight_0"+ str(dfg+1)+"/f0"+str(dfg+1)+"_hr_tif/"
#child2_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cmb/flight_0"+ str(dfg+1)+"/f0"+str(dfg+1)+"_hr_tif/"
#child2_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cm_hn/flight_0"+ str(dfg+1)+"/f0"+str(dfg+1)+"_hr_tif/"
child2_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cm_30m/flight_0"+ str(dfg+1)+"/f0"+str(dfg+1)+"_hr_tif/"
os.makedirs(child2_folder_path, exist_ok=True)
#child3_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs/flight_0"+ str(dfg+1)+"/f0"+str(dfg+1)+"_lr_tif/"
#child3_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_2/flight_0"+ str(dfg+1)+"/f0"+str(dfg+1)+"_lr_tif/"
#child3_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cmb/flight_0"+ str(dfg+1)+"/f0"+str(dfg+1)+"_lr_tif/"
#child3_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cm_hn/flight_0"+ str(dfg+1)+"/f0"+str(dfg+1)+"_lr_tif/"
child3_folder_path = r"D:/work/research_t/downscaling/pre_proc_imgs_3x3cm_30m/flight_0"+ str(dfg+1)+"/f0"+str(dfg+1)+"_lr_tif/"
os.makedirs(child3_folder_path, exist_ok=True)
# Assigning folders
it_01_square_pol_folder = child1_folder_path
it_02_HR_tif_folder = child2_folder_path
it_03_LR_tif_folder = child3_folder_path

#
of_01_buffer  = father_folder_path + "/f0" + str(dfg+1) + "_buffer.shp"
of_02_rdm_pts = father_folder_path + "/f0" + str(dfg+1) + "_random_pts.shp"
of_03_rdm_sqs = father_folder_path + "/f0" + str(dfg+1) + "_random_sqs.shp"
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------
### --- Set Buffer over the UAV-flight paths and save as shp file

# Load the shapefile
input_shapefile = if_01_border_file

# buffered_shapefile = "D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_buff.shp"
buffered_shapefile = of_01_buffer

gdf = gpd.read_file(input_shapefile)

#buffer_distance = -40  # Adjust this distance               <---------- SET
buffer_distance = -20  # Adjust this distance               <---------- SET
gdf["geometry"] = gdf.geometry.buffer(buffer_distance)

### Save the new shapefile with buffer
gdf.to_file(buffered_shapefile)
print("Buffered shapefile saved successfully!")
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------
### Generate points each 3 m & save it as SHP file (like fishnet)
start_time = time.perf_counter ()  

# Get the bounding box (minx, miny, maxx, maxy)
#------ shapefile_path = "D:/work/research_t/downscaling/ForEfrain/borders/border_04_S1.shp"
shapefile_path = if_01_border_file
gdf = gpd.read_file(shapefile_path)

minx, miny, maxx, maxy = gdf.total_bounds
print("Extent of the shapefile:")
print(f"Min X: {minx}, Min Y: {miny}")
print(f"Max X: {maxx}, Max Y: {maxy}")

x1, y1 = 413910, 3407277         # Define the initial coordinates of large Frame 
cell_size_x, cell_size_y = 3, 3  # Define spacing between points

nx1 = int((minx-x1)/3)
nx2 = int((maxx-x1)/3)
ny1 = int((miny-y1)/3)
ny2 = int((maxy-y1)/3)
#print(nx1, nx2, ny1, ny2)

xmin, xmax = x1 + 3*nx1, x1 + 3*nx2
ymin, ymax = y1 + 3*ny1, y1 + 3*ny2
print(f"Min X: {xmin}, Min Y: {ymin}")
print(f"Max X: {xmax}, Max Y: {ymax}")

# Generate grid coordinates
x_coords = np.arange(xmin, xmax + cell_size_x, cell_size_x)
y_coords = np.arange(ymin, ymax + cell_size_y, cell_size_y)
points = [Point(x, y) for x in x_coords for y in y_coords]   # Create points

# Create GeoDataFrame
fishnet_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:32614")

# Save to Shapefile of generated points
#fishnet_gdf.to_file("D:/work/research_t/downscaling/ForEfrain/points/t_fishnet_01.shp")
print("Fishnet grid of points created in memory")#" and saved as 'fishnet_points.shp'")
print("------> Running Time: ", time.perf_counter() - start_time, "seconds")
# ------> Running Time:  6.30373439998948 seconds
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
### Clip Fishnet points
start_time = time.perf_counter ()
# Load the point shapefile & polygon shapefile 
# buff_polyg = gpd.read_file("D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_buff.shp")
buff_polyg = gpd.read_file(of_01_buffer)

# Ensure both shapefiles have the same CRS
fishnet_gdf = fishnet_gdf.to_crs(buff_polyg.crs)

# Clip the points with the polygon
# clipped_points = gpd.clip(fishnet_points, polygon)
clipped_points = gpd.clip(fishnet_gdf, buff_polyg) 

# Save the output
#clipped_points.to_file("D:/work/research_t/downscaling/ForEfrain/points/t2_clip_pts_01.shp")

print("Clipping completed! Clipped points saved.")
print("------> Running Time: ", time.perf_counter() - start_time, "seconds")
# ------> Running Time:  1.424132300002384 seconds
# -----------------------------------------------------------------------------







# -----------------------------------------------------------------------------
#### Selection of random points

### ---------- This code uses a file saved in memory (fishnet is in Memory - No in disk)
# --- Selection of Random Points ---
def subset_points_from_gdf(gdf, num_points, min_distance):
    """
    Select a subset of points from a GeoDataFrame such that each selected point 
    is at least `min_distance` apart. Returns a new GeoDataFrame with the selected points.
    """
    print(num_points, min_distance)
    # Extract point coordinates
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    
    # Randomly shuffle coordinates (using a fixed seed for reproducibility)
    np.random.seed(50)
    np.random.shuffle(coords)
    
    selected_points = []
    for point in coords:
        if not selected_points:
            selected_points.append(point)
        else:
            tree = cKDTree(selected_points)
            min_dist, _ = tree.query(point, k=1)
            if min_dist >= min_distance:
                selected_points.append(point)
        if len(selected_points) >= num_points:
            break
        
    # Save the subset as a new shapefile
    #output_shapefile = "D:/work/research_t/downscaling/pre_proc_imgs_3x3cm_30m/flight_01/sle_ptns.shp"
    #selected_gdf.to_file(output_shapefile, driver="ESRI Shapefile")
    #print(f"Subset saved to {output_shapefile}, with {len(selected_points)} points.")

    # Convert the selected points to a GeoDataFrame
    return gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in selected_points],
        crs=gdf.crs
    )

# Call the function using the in-memory clipped_points GeoDataFrame
# Setting small 'min_distance' may generate too much overlapping between square polygons
#selected_gdf = subset_points_from_gdf(clipped_points, num_points=1500, min_distance=40)
#selected_gdf = subset_points_from_gdf(clipped_points, num_points=1500, min_distance=50)
selected_gdf = subset_points_from_gdf(clipped_points, num_points=1500, min_distance=25) # <------- SET

# Save the selected points (if needed)
#rdm_pts_shp = "D:/work/research_t/downscaling/ForEfrain/points/f01_rdm_pts.shp"
#------ rdm_pts_shp = "D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_rdm_pts.shp"
rdm_pts_shp = of_02_rdm_pts
selected_gdf.to_file(rdm_pts_shp, driver="ESRI Shapefile")
print(colored(220, 200, 0, ("Random point selection completed!", len(selected_gdf))))  #'''
print("------> Running Time:", time.perf_counter() - start_time, "seconds")
# ------> Running Time: 215.05878850000045 seconds
# ------> Running Time: 640.4895418000015 seconds
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
### Generate squares
def generate_squares(input_shapefile, output_shapefile, square_size):
    # Load point shapefile
    gdf = gpd.read_file(input_shapefile)
    
    # Compute half-size for centering the square around the point
    half_size = square_size / 2
    
    # Generate squares around each point
    squares = []
    for point in gdf.geometry:
        x, y = point.x, point.y
        square = Polygon([
            (x - half_size, y - half_size),  # Bottom-left
            (x - half_size, y + half_size),  # Top-left
            (x + half_size, y + half_size),  # Top-right
            (x + half_size, y - half_size)   # Bottom-right
        ])
        squares.append(square)

    # Create new GeoDataFrame for squares
    square_gdf = gpd.GeoDataFrame(geometry=squares, crs=gdf.crs)

    # Save to a new shapefile
    square_gdf.to_file(output_shapefile, driver="ESRI Shapefile")
    print(f"Squares saved to {output_shapefile}")

# Example Usage
#input_shapefile = "D:/work/research_t/downscaling/ForEfrain/points/D01c_subset_pts_a.shp"  # Replace with your actual file
#output_shapefile = "D:/work/research_t/downscaling/ForEfrain/points/D01c_subset_sq_a.shp" # <--- SET 1
#------ input_shapefile = "D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_rdm_pts.shp"  # Replace with your actual file
input_shapefile = of_02_rdm_pts

#rdm_squares_shp = "D:/work/research_t/downscaling/ForEfrain/points/f01_rdm_sq.shp" # <--- SET 1
#------ rdm_squares_shp = "D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_rdm_sqs.shp" # <--- SET 1
rdm_squares_shp = of_03_rdm_sqs

#generate_squares(input_shapefile, rdm_squares_shp, square_size=60)
generate_squares(input_shapefile, rdm_squares_shp, square_size=30)           # < ----------SET
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------
### Create a POLYGON shapefile for each square polygon
def split_polygons(input_shapefile, output_folder):
    # Load the shapefile
    gdf = gpd.read_file(input_shapefile)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over each polygon and save it separately
    list_shp = []
    for idx, row in gdf.iterrows():
        single_gdf = gpd.GeoDataFrame([row], columns=gdf.columns, crs=gdf.crs)
        output_shapefile = os.path.join(output_folder, f"polygon_{idx}.shp")
        list_shp.append(f"polygon_{idx}.shp")
        single_gdf.to_file(output_shapefile, driver="ESRI Shapefile")
        print(f"Saved {output_shapefile}")
    return list_shp

# Example Usage
#input_shapefile = "D:/work/research_t/downscaling/ForEfrain/points/D01c_subset_sq_a.shp" # <--- SET 1
#------ input_shapefile = "D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_rdm_sqs.shp"
input_shapefile = of_03_rdm_sqs

#dron_f_path = "D:/work/research_t/downscaling/s2_folder"
#dron_f_path = "D:/work/research_t/downscaling/D01c/D01_polys"                           # <--- SET 1
#------ dron_f_path = "D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_polyg"
dron_f_path = it_01_square_pol_folder

os.makedirs(dron_f_path, exist_ok=True)  # Creates folder if it doesn’t exist
print(f"Directory '{dron_f_path}' created successfully.")

list_pol_files = split_polygons(input_shapefile, dron_f_path)
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------
### ------------- Clip HR tif using polygons
def clip_raster2(raster_path, shp_f, in_path, out_path, expected_size, pixel_size):
    # Read the polygon geometry from the shapefile
    shp_f_full_name = os.path.join(in_path, shp_f)
    #out_tif_f = out_path + shp_f.replace(".shp", ".tif")
    #("------>", shp_f_full_name)
    #print("------>", out_tif_f)
    with fiona.open(shp_f_full_name, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        # Get the overall bounds of the shapefile (assuming it contains one polygon)
        poly_bounds = shapefile.bounds  # (minx, miny, maxx, maxy)
        
    # Unpack bounds; note that in most projected CRSs, y increases upward.
    left, bottom, right, top = poly_bounds
    
    # Open the raster file and clip it using the polygon
    with rasterio.open(raster_path) as src:
        clipped_image, clipped_transform  = mask(src, shapes, crop=True)
        src_crs = src.crs
        src_dtype = clipped_image.dtype
    
    # Save the intermediate clipped raster (optional)
    clipped_meta = src.meta.copy()
    clipped_meta.update({
        "driver": "GTiff",
        "height": clipped_image.shape[1],
        "width": clipped_image.shape[2],
        "transform": clipped_transform
    })
    ### Save the tif (This tif has no-fixed bounds)
    '''
    clipped_out_tif = out_path + shp_f.replace(".shp", "e.tif")
    with rasterio.open(clipped_out_tif, "w", **clipped_meta) as dst:
        dst.write(clipped_image)
    #print("Clipped raster saved at:", clipped_out_tif)  #'''
    
    
    # --- Step 2: Resample to Exact 2000x2000 Pixels ---
    # Desired parameters:
    #target_width = 2000
    #target_height = 2000
    # --------------pixel_size = 0.03  # each pixel will be exactly 0.03 x 0.03 m
    #pixel_size = 0.03  # each pixel will be exactly 0.3 x 0.3 m
    
    # If the polygon is exactly 60m x 60m:
    target_width = int(round((right - left) / pixel_size))   # Should be 2000
    target_height = int(round((top - bottom) / pixel_size))    # Should be 2000
    #print("Expected dimensions:", target_width, "x", target_height)
    
    # Define a new transform using the upper-left coordinates from the clipped transform.
    # Here, we assume that the clipped_transform.c and clipped_transform.f represent the origin.
    #new_transform = rasterio.Affine(
    #    pixel_size, 0, clipped_transform.c,
    #    0, -pixel_size, clipped_transform.f
    #)
    new_transform = rasterio.Affine(pixel_size, 0, left,
                                0, -pixel_size, top)
    
    # Create an empty array to hold the resampled data
    # (Number of bands, target_height, target_width)
    #dest_array = np.empty((clipped_image.shape[0], target_height, target_width), dtype=src_dtype)
    dest_array = np.empty((clipped_image.shape[0], target_height, target_width), dtype=src_dtype)

    
    # Reproject/resample the clipped image to the new grid
    reproject(
        source=clipped_image,
        destination=dest_array,
        src_transform=clipped_transform,
        src_crs=src_crs,
        dst_transform=new_transform,
        dst_crs=src_crs,
        resampling=Resampling.nearest,  # you can choose another method if needed
        dst_nodata=None  # Ensures no extra nodata is forced in
    )
    
    # Update metadata for the resampled raster
    resampled_meta = clipped_meta.copy()
    resampled_meta.update({
        "height": target_height,
        "width": target_width,
        "transform": new_transform
    })
    
    ### Save the clipped raster to a new file
    clipped_out_fixed_tif = out_path + shp_f.replace(".shp", ".tif")
    with rasterio.open(clipped_out_fixed_tif, "w", **resampled_meta) as dest:
        dest.write(dest_array) #'''
    #print("Raster has been clipped and saved successfully!")
            
    return clipped_out_fixed_tif #'''


def process_rasters(raster_path, list_pol_files, in_path, out_path, expected_size, pixel_size):
    #process_rasters(raster_path, list_pol_files, square_pol_folder, hr_tif_path)
    list_tifs = []
    for shp_f in tqdm(list_pol_files, desc="Processing Shapefiles", unit="file"):
        #out_tif_f = clip_raster(raster_path, shp_f, in_path, out_path, expected_size)
        out_tif_f = clip_raster2(raster_path, shp_f, in_path, out_path, expected_size, pixel_size)
        list_tifs.append(out_tif_f)
    return list_tifs

# Paths to your files
#HR_raster_img = "D:/work/research_t/downscaling/ForEfrain/wf_p_resampled/01_E1_r4.tif"
#HR_raster_img = "D:/work/research_t/downscaling/ForEfrain/wf_p_resampled/04_S1_r.tif"
#------ HR_raster_img = "D:/work/research_t/downscaling/ForEfrain/wf_p_resampled/03_NE_r.tif"
HR_raster_img = if_02_HR_raster_img

#square_pol_folder = "D:/work/research_t/downscaling/s2_folder/"
#square_pol_folder = "D:/work/research_t/downscaling/D01c/D01_polys"               # <--- SET 1
#------ square_pol_folder = "D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_polyg"
square_pol_folder = it_01_square_pol_folder

#HR_tif_folder = "D:/work/research_t/downscaling/s3_hr_tif/"
#HR_tif_folder = "D:/work/research_t/downscaling/D01c/D01_hr_tif/"                  # <--- SET 1
#------ HR_tif_folder = "D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_hr_tif/"  # <--- SET 1
HR_tif_folder = it_02_HR_tif_folder
os.makedirs(HR_tif_folder, exist_ok=True)  # Creates folder if it doesn't exist
print(f"Directory '{HR_tif_folder}' created successfully.")

# Process rasters and filter valid ones
start_time = time.perf_counter()

#list_pol_files = []  # List your shapefiles here
# --------------- exp_size = 2000 # 2500 # 3000 # 2000          60/0.03 = 2000            # <--- SET 2
exp_size = 2000 #                                     60/0.3 = 200            # <--- SET 2
pixel_size = 0.03
list_tifs = process_rasters(HR_raster_img, list_pol_files, square_pol_folder, HR_tif_folder, exp_size, pixel_size)
print(colored(255, 255, 0, ("------> N_squares processed: "+ str(len(list_tifs)))))


# ---
### Filter HR files only with data (Tiles with no-data are deleted)

def filter_valid_lr_tifs(list_tifs):
    valid_tifs = []
    
    for tif_f in list_tifs:
        with rasterio.open(tif_f) as src:
            data_b1 = src.read(1)
            nodata_value_b1 = src.nodata
            nodata_pixels_b1 = np.sum(data_b1 == nodata_value_b1) if nodata_value_b1 is not None else 0
            
            data_b2 = src.read(2)
            nodata_value_b2 = src.nodata
            nodata_pixels_b2 = np.sum(data_b2 == nodata_value_b2) if nodata_value_b2 is not None else 0
            
            data_b3 = src.read(3)
            nodata_value_b3 = src.nodata
            nodata_pixels_b3 = np.sum(data_b3 == nodata_value_b3) if nodata_value_b3 is not None else 0
            
            data_b4 = src.read(4)
            nodata_value_b4 = src.nodata
            nodata_pixels_b4 = np.sum(data_b4 == nodata_value_b4) if nodata_value_b4 is not None else 0
            
            valid_pixels_b1 = data_b1[data_b1 != src.nodata]
            min_val_b1, max_val_b1 = np.min(valid_pixels_b1), np.max(valid_pixels_b1)
            if max_val_b1 >= 1:
                print(f"POL: {os.path.splitext(os.path.basename(tif_f))[0]}  Min val_B1: {min_val_b1}   Max val_B1: {max_val_b1}")
            
            valid_pixels_b2 = data_b2[data_b2 != src.nodata]
            min_val_b2, max_val_b2 = np.min(valid_pixels_b2), np.max(valid_pixels_b2)
            if max_val_b2 >= 1:
                print(f"POL: {os.path.splitext(os.path.basename(tif_f))[0]}  Min val_B2: {min_val_b2}   Max val_B2: {max_val_b2}")
            
            valid_pixels_b3 = data_b3[data_b3 != src.nodata]
            min_val_b3, max_val_b3 = np.min(valid_pixels_b3), np.max(valid_pixels_b3)
            if max_val_b3 >= 1:
                print(f"POL: {os.path.splitext(os.path.basename(tif_f))[0]}  Min val_B3: {min_val_b3}   Max val_B3: {max_val_b3}")
            
            valid_pixels_b4 = data_b4[data_b4 != src.nodata]
            min_val_b4, max_val_b4 = np.min(valid_pixels_b4), np.max(valid_pixels_b4)
            if max_val_b4 >= 1:
                print(f"POL: {os.path.splitext(os.path.basename(tif_f))[0]}  Min val_B4: {min_val_b4}   Max val_B4: {max_val_b4}")
        
        #if nodata_pixels_b1 == 0:
        if nodata_pixels_b1 == 0 and nodata_pixels_b2 == 0 and nodata_pixels_b3 == 0 and nodata_pixels_b4 == 0:
            valid_tifs.append(os.path.splitext(os.path.basename(tif_f))[0])
        else:
            pol_del = os.path.splitext(os.path.basename(tif_f))[0]
            print(colored(255, 255, 0, (f"Deleting {pol_del} due to NoData pixels")))  #'''
            #print("NoData B1: ", nodata_pixels_b1)
            #print("NoData B2: ", nodata_pixels_b2)
            #print("NoData B3: ", nodata_pixels_b3)
            #print("NoData B4: ", nodata_pixels_b4)
            print("NoData B1, B2, B3, B4: ", nodata_pixels_b1, nodata_pixels_b2, nodata_pixels_b3, nodata_pixels_b4)
            os.remove(tif_f)  # Remove tif file wit NoData
    return valid_tifs

tif_aproved = filter_valid_lr_tifs(list_tifs)
#-------------- tif_aproved.remove('polygon_61')
print(colored(0, 255, 255, ('Aproved tif files: ' + str(len(tif_aproved))+' out '+str(len(list_tifs)))))
print("------> Running Time: ", time.perf_counter() - start_time, "seconds")
# ------> Running Time:  160.8749723000219 seconds
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
### ------------- Clip LR tif
start_time = time.perf_counter()
#------ LR_raster_img = "D:/work/research_t/downscaling/ForEfrain/Planet_select/01_2023-04-03_psscene_analytic_sr_udm2/composite.tif"
LR_raster_img = if_03_LR_raster_img
#square_pol_folder = "D:/work/research_t/downscaling/s2_folder/"
#square_pol_folder = "D:/work/research_t/downscaling/D01c/D01_polys"            # <--- SET 1
#------ square_pol_folder = "D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_polyg"
square_pol_folder = it_01_square_pol_folder

#LR_tif_folder = "D:/work/research_t/downscaling/s3_lr_tif/"
#LR_tif_path = "D:/work/research_t/downscaling/D01c/D01_lr_tif/"                # <--- SET 1
#------ LR_tif_folder = "D:/work/research_t/downscaling/pre_proc_imgs/flight_04/f04_lr_tif/"  # <--- SET 1
LR_tif_folder = it_03_LR_tif_folder
os.makedirs(LR_tif_folder, exist_ok=True)  # Creates both parent_folder and child_folder if they don’t exist
print(f"Directory '{LR_tif_folder}' created successfully.")

list_pol_aprov_files = [f"{p}.shp" for p in tif_aproved]
exp_size = 20 #30 # 20                    60/3 = 20
pixel_size = 3
list_lr_tifs = process_rasters(LR_raster_img, list_pol_aprov_files, square_pol_folder, LR_tif_folder, exp_size, pixel_size)

print("------> Running Time: ", time.perf_counter() - start_time, "seconds")
# ------> Running Time:  1.56105119996937 seconds
# -----------------------------------------------------------------------------






# -----------------------------------------------------------------------------
### Check dimension, number of piixels, file size, cells with and wit no data
'''
import rasterio
import os

# Path to your TIFF file
tif_path = r"D:\work\research_t\downscaling\pre_proc_imgs\flight_03\f03_hr_tif\polygon_1.tif"
#tif_path = r"D:\work\research_t\downscaling\pre_proc_imgs\flight_03\f03_hr_tif\polygon_0r.tif"

# Get the file size in bytes
file_size_bytes = os.path.getsize(tif_path)
file_size_mb = file_size_bytes / (1024 * 1024)

# Open the raster file with rasterio
with rasterio.open(tif_path) as src:
    width = src.width
    height = src.height
    pixel_count = width * height  # Total number of pixels
    print(f"Raster dimensions: {width} x {height}")
    print(f"Total number of pixels: {pixel_count}")

print(f"File size: {file_size_bytes} bytes (~{file_size_mb:.2f} MB)")


with rasterio.open(tif_path) as src:
    # Read the first band
    band = src.read(1)
    nodata = src.nodata

    if nodata is None:
        print("No nodata value defined. Assuming all pixels have valid data.")
        count_data = band.size
        count_nodata = 0
    else:
        # Count pixels with no data
        count_nodata = np.count_nonzero(band == nodata)
        # Count pixels with data
        count_data = np.count_nonzero(band != nodata)

    print(f"Pixels with data: {count_data}")
    print(f"Pixels with no data: {count_nodata}")
# '''
# -----------------------------------------------------------------------------























































