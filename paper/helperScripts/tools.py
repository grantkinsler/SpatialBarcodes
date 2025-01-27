import numpy as np
# import cv2
import tifffile as tiff
from shapely.geometry import Polygon, mapping, shape, box
from skimage import io
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
# import json
from rasterio import features
import scanpy as sc
import pandas as pd
import geopandas as gpd

from ast import literal_eval

from matplotlib import colors

import pickle
import scipy

import seaborn as sns

from SGanalysis.SGobject import SGobject

import matplotlib
# %matplotlib inline
# matplotlib.use('Qt5Agg')
# %matplotlib qt

roi_file_paths = {

    # 'roi_1':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-05-21_mouseexp_expression/projects/roi_1/segmentations/nuclei_20240604_nuclei.tiff',
    #                        'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-05-21_mouseexp_expression/projects/roi_1/exports/decode_20240604.csv',
    #                         'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-05-21_mouseexp_expression/projects/roi_1/exports'
    #                        },
                  'roi_2':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-05-21_mouseexp_expression/projects/roi_2/segmentations/nuclei_20240529_nuclei.tiff',
                           'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-05-21_mouseexp_expression/projects/roi_2/exports/decode_20240604.csv',
                            'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-05-21_mouseexp_expression/projects/roi_2/exports'
                           },
    #               'roi_3':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-05-21_mouseexp_expression/projects/roi_3/segmentations/nuclei_20240604_nuclei.tiff',
    #                        'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-05-21_mouseexp_expression/projects/roi_3/exports/decode_20240604.csv',
    #                         'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-05-21_mouseexp_expression/projects/roi_3/exports'
    #                        },
                  # 'timezero_roi_1':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-04-27_spatialbarcodes_SG_expression_mouse_exp/time_zero_output/roi1/segmentations/segmentation_20240513_nuclei.tiff',
                  #             'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-04-27_spatialbarcodes_SG_expression_mouse_exp/time_zero_output/roi1/transcripts/20240517_segmentation_withRefid.csv',
                  #             'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-04-27_spatialbarcodes_SG_expression_mouse_exp/time_zero_output/roi1/exports'
                  #             },
                  #   'timezero_roi_2':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-04-27_spatialbarcodes_SG_expression_mouse_exp/time_zero_output/roi2/segmentations/segment_091924_nuclei.tiff',
                  #             'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-04-27_spatialbarcodes_SG_expression_mouse_exp/time_zero_output/roi2/transcripts/transcripts.csv',
                  #             'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-04-27_spatialbarcodes_SG_expression_mouse_exp/time_zero_output/roi2/exports'
                  #             },
                  #   'timezero_roi_3':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-04-27_spatialbarcodes_SG_expression_mouse_exp/time_zero_output/roi3/segmentations/segment_20240921_nuclei.tiff',
                  #             'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-04-27_spatialbarcodes_SG_expression_mouse_exp/time_zero_output/roi3/transcripts/transcripts.csv',
                  #             'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-04-27_spatialbarcodes_SG_expression_mouse_exp/time_zero_output/roi3/exports'
                  #             },


                   #   'run2_roi_1':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-08-08_spatialbarcode_tumor2_projects/roi_1/exports/segmentation.tiff',
                   #            'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-08-08_spatialbarcode_tumor2_projects/roi_1/exports/transcripts.csv',
                   #             'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-08-08_spatialbarcode_tumor2_projects/roi_1/exports/',
                   #              },
                   # 'run2_roi_2':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-08-08_spatialbarcode_tumor2_projects/roi_2/exports/segmentation.tiff',
                   #             'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-08-08_spatialbarcode_tumor2_projects/roi_2/exports/transcripts.csv',
                   #             'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-08-08_spatialbarcode_tumor2_projects/roi_2/exports/',
                   #             },
                   # 'run2_roi_3':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-08-08_spatialbarcode_tumor2_projects/roi_3/exports/segmentation.tiff',
                   #             'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-08-08_spatialbarcode_tumor2_projects/roi_3/exports/transcripts.csv',
                   #             'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-08-08_spatialbarcode_tumor2_projects/roi_3/exports/',
                   #             },


                   #  #### for running on McRyanFace
                   #  'run2_roi_1':{'segmentation_file':'/media/wniu/GRK_003/2024-08-08_spatialbarcode_tumor2_projects/roi_1/exports/segmentation.tiff',
                   #            'spots_file':'/media/wniu/GRK_003/2024-08-08_spatialbarcode_tumor2_projects/roi_1/exports/transcripts.csv',
                   #             'out_path':'/media/wniu/GRK_003/2024-08-08_spatialbarcode_tumor2_projects/roi_1/exports/',
                   #              },
                   # 'run2_roi_2':{'segmentation_file':'/media/wniu/GRK_003/2024-08-08_spatialbarcode_tumor2_projects/roi_2/exports/segmentation.tiff',
                   #             'spots_file':'/media/wniu/GRK_003/2024-08-08_spatialbarcode_tumor2_projects/roi_2/exports/transcripts.csv',
                   #             'out_path':'/media/wniu/GRK_003/2024-08-08_spatialbarcode_tumor2_projects/roi_2/exports/',
                   #             },
                   # 'run2_roi_3':{'segmentation_file':'/media/wniu/GRK_003/2024-08-08_spatialbarcode_tumor2_projects/roi_3/exports/segmentation.tiff',
                   #             'spots_file':'/media/wniu/GRK_003/2024-08-08_spatialbarcode_tumor2_projects/roi_3/exports/transcripts.csv',
                   #             'out_path':'/media/wniu/GRK_003/2024-08-08_spatialbarcode_tumor2_projects/roi_3/exports/',
                   #             },
                    'dish_roi1':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-02-27_spatialbarcodes_SG_expression/projects/2024-02-27_spatialbarcodes_expression/roi_1/segmentations/segmentation_1_nuclei.tiff',
                              'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-02-27_spatialbarcodes_SG_expression/projects/2024-02-27_spatialbarcodes_expression/roi_1/exports/transcripts.csv',
                              'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-02-27_spatialbarcodes_SG_expression/projects/2024-02-27_spatialbarcodes_expression/roi_1/exports'
                              },
        
                    'dish_roi2':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-02-27_spatialbarcodes_SG_expression/projects/2024-02-27_spatialbarcodes_expression/roi_2/segmentations/segmentation_1_nuclei.tiff',
                              'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-02-27_spatialbarcodes_SG_expression/projects/2024-02-27_spatialbarcodes_expression/roi_2/exports/transcripts.csv',
                              'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-02-27_spatialbarcodes_SG_expression/projects/2024-02-27_spatialbarcodes_expression/roi_2/exports/'
                              },

                    'dish_roi3':{'segmentation_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-02-27_spatialbarcodes_SG_expression/projects/2024-02-27_spatialbarcodes_expression/roi_3/segmentations/20240321_segmentation_2_nuclei.tiff',
                              'spots_file':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-02-27_spatialbarcodes_SG_expression/projects/2024-02-27_spatialbarcodes_expression/roi_3/exports/transcripts.csv',
                              'out_path':'/Users/grantkinsler/RajLab Dropbox/Grant Kinsler/SpatialBarcodes/ImagingData/2024-02-27_spatialbarcodes_SG_expression/projects/2024-02-27_spatialbarcodes_expression/roi_3/exports/'
                              },
                  
                  }



# this function jitters a point by the standard deviation
# useful for jittering points for visualization
def jitter_point(mean,std=0.15):
    return np.random.normal(mean,std)

# This function retrieves the polygons around a given polygon in an SGobject.
# It takes the SGobject, identifier, id_field, and image_scale as input parameters.
def get_polygons_around_polygon(sg_obj, identifier, id_field='object_id', image_scale=5):

    # Retrieve the polygon geometry for the given identifier
    polygon_gdf = sg_obj.gdf[sg_obj.gdf[id_field] == identifier]

    # Get the bounds of the polygon geometry
    minx, miny, maxx, maxy = polygon_gdf.geometry.iloc[0].bounds

    # Calculate the expansion factor for the bounding box
    dx = (maxx - minx) * 0.5 * image_scale
    dy = (maxy - miny) * 0.5 * image_scale

    # Create an expanded bounding box around the polygon
    expanded_bbox = box(minx - dx, miny - dy, maxx + dx, maxy + dy)

    # Retrieve all other polygons that intersect with the expanded bounding box (excluding the focal polygon)
    other_polygons = sg_obj.gdf[sg_obj.gdf.geometry.intersects(expanded_bbox) & (sg_obj.gdf[id_field] != identifier)]

    # Return the other polygons
    return other_polygons

def get_polygon_barcodes(df,identifier,id_field='cell_id',literal_eval_bool=False):

    if len(df[df[id_field]==identifier]['called_barcodes'].values) > 0:
        if literal_eval_bool:
            return literal_eval(df[df[id_field]==identifier]['called_barcodes'].values[0])
        else:
            return df[df[id_field]==identifier]['called_barcodes'].values[0]

    else:
        return []

def get_all_barcodes_in_region(df,identifiers,id_field='cell_id',literal_eval_bool=False):

    bc_list = []

    for identifier in identifiers:
        bc_list += get_polygon_barcodes(df,identifier,id_field=id_field,literal_eval_bool=literal_eval_bool)

    return bc_list


def get_barcodes_within_polygon(df,identifier,id_field='cell_id',barcode_cols=[f'bc_{i:03}' for i in range(1,97)]):

    this_polygon = df[df[id_field]==identifier]
    m2 = (this_polygon[barcode_cols] > 0).any()
    cols = m2.index[m2].tolist()
    return cols

def get_barcodes_within_polygonlist(df,identifier_list,id_field='cell_id',barcode_cols=[f'bc_{i:03}' for i in range(1,97)]):

    this_polygon = df[df[id_field].isin(identifier_list)]
    m2 = (this_polygon[barcode_cols] > 0).any()
    cols = m2.index[m2].tolist()
    return cols





def show_neighborhood(sg_obj, identifier, image_path, id_field='object_id', image_scale=1.0,ax=None):
    #   transcripts, gene, gfp_coords,self_expression, nn_mean, output_figure_path, figure_size=[],width_cm = 8, height_cm = 8 ,dpi=300):
    """
    Display the neighborhood of a specific object in an image.

    Parameters:
    - sg_obj: SGobject containing the spatial information.
    - identifier: Identifier of the object to display the neighborhood for.
    - image_path: Path to the TIFF image file.
    - id_field: Field name in the sg_obj.gdf DataFrame that contains the object identifiers. Default is 'object_id'.
    - image_scale: Scale factor to adjust the size of the neighborhood. Default is 1.0.

    Returns:
    - fig: Matplotlib figure object displaying the neighborhood.

    Note:
    - This function requires the following libraries: tiff, box, np, plt from matplotlib.

    Example usage:
    ```
    sg_obj = SGobject(...)
    identifier = 'object123'
    image_path = '/path/to/image.tif'
    fig = show_neighberhood(sg_obj, identifier, image_path)
    plt.show()
    ```
    """
    # Load the TIFF file
    image = tiff.imread(image_path)

    polygon_gdf = sg_obj.gdf[sg_obj.gdf[id_field] == identifier]
    first_polygon_geometry = polygon_gdf.geometry.iloc[0]

    minx, miny, maxx, maxy = first_polygon_geometry.bounds
    dx = (maxx - minx) * 0.5 * image_scale
    dy = (maxy - miny) * 0.5 * image_scale
    expanded_bbox = box(minx - dx, miny - dy, maxx + dx, maxy + dy)

    # get other polygons in this region
    # other_polygons = sg_obj.gdf[sg_obj.gdf.geometry.intersects(expanded_bbox) & (sg_obj.gdf[id_field] != identifier)]

    # specify the range of the image to display
    range_y_lower = int(miny - dy)
    range_y_upper = int(maxy + dy)
    range_x_lower = int(minx - dx)
    range_x_upper = int(maxx + dx)

    # Ensure the ranges are within image bounds
    range_y_lower = max(range_y_lower, 0)
    range_y_upper = min(range_y_upper, image.shape[1])
    range_x_lower = max(range_x_lower, 0)
    range_x_upper = min(range_x_upper, image.shape[2])

    # Create the boolean mask
    # spot_in_range = ((spot_coords[:, 1] > range_y_lower) & (spot_coords[:, 1] < range_y_upper)) & \
    #                 ((spot_coords[:, 0] > range_x_lower) & (spot_coords[:, 0] < range_x_upper))
    # print(np.shape(spot_in_range))

    # Crop the fourth channel using the bounding rectangle
    cropped_image = image[3, range_y_lower:range_y_upper, range_x_lower:range_x_upper]

    # Check if cropped_image is non-empty
    if cropped_image.size == 0:
        print(f"No pixels found in the specified range for coordinates {coordinates}")
        return

    gray_array = np.array(cropped_image)

    # Normalize the pixel values to the range 0-255
    normalized_gray = (gray_array - gray_array.min()) / (gray_array.max() - gray_array.min()) * 255
    normalized_gray = normalized_gray.astype(np.uint8)

    # Increase the saturation of the blue channel
    saturated_blue = normalized_gray * 1.5  # Increase intensity by 50%
    saturated_blue = np.clip(saturated_blue, 0, 255).astype(np.uint8)  # Ensure values are within 0-255

    # Create a new image with the blue channel based on the saturated grayscale values
    blue_image_array = np.zeros((gray_array.shape[0], gray_array.shape[1], 3), dtype=np.uint8)
    blue_image_array[:, :, 2] = saturated_blue  # Set the blue channel
    # also set other channels
    blue_image_array[:, :, 0] = saturated_blue  # Set the blue channel
    blue_image_array[:, :, 1] = saturated_blue  # Set the blue channel


    blue_image_array[:, :, 2] = normalized_gray  # Set the blue channel
    # also set other channels
    blue_image_array[:, :, 0] = normalized_gray   # Set the blue channel
    blue_image_array[:, :, 1] = normalized_gray  # Set the blue channel

    # Create a figure
    fig, ax = plt.subplots()
    ax.imshow(blue_image_array,extent=[range_x_lower,range_x_upper,range_y_lower,range_y_upper])

    polygon_gdf.boundary.plot(ax=ax, color='red', linewidth=2)
    # other_polygons.boundary.plot(ax=ax, color='w', linewidth=1)


    ax.axis('off')  # Hide axis
    # ax.scatter(spot_coords[spot_in_range, 0] - range_x_lower, spot_coords[spot_in_range, 1] - range_y_lower, c='r', s=1)
    # ax.scatter(coordinates[:, 0] - range_x_lower, coordinates[:, 1] - range_y_lower, c='m', s=30)
    # ax.scatter(gfp_coords[0] - range_x_lower, gfp_coords[1] - range_y_lower, c='g', s=30)
    # width_inch = width_cm / 2.54
    # height_inch = height_cm / 2.54
    # fig.set_size_inches(width_inch, height_inch)
    # Add text annotations using relative coordinates
    # ax.text(0.1, 0.9, 'sprinkled cell ' + gene + ' expression: ' + str(np.round( self_expression, decimals=1)), transform=ax.transAxes, color='white', fontsize=10, ha='left', va='center')
    # ax.text(0.1, 0.82, 'nearest neighbors mean ' + gene + ' expression: ' + str(np.round(nn_mean, decimals=1)), transform=ax.transAxes, color='white', fontsize=10,
    #         ha='left', va='center')

    # Save the figure with specified resolution
    # plt.savefig(output_figure_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    # plt.close()
    # return fig

def show_neighborhood_subimage(image_path,bounds_tuple,ax=None):
    #   transcripts, gene, gfp_coords,self_expression, nn_mean, output_figure_path, figure_size=[],width_cm = 8, height_cm = 8 ,dpi=300):
    """
    Display the neighborhood of a specific object in an image.

    Parameters:
    - sg_obj: SGobject containing the spatial information.
    - identifier: Identifier of the object to display the neighborhood for.
    - image_path: Path to the TIFF image file.
    - id_field: Field name in the sg_obj.gdf DataFrame that contains the object identifiers. Default is 'object_id'.
    - image_scale: Scale factor to adjust the size of the neighborhood. Default is 1.0.

    Returns:
    - fig: Matplotlib figure object displaying the neighborhood.

    Note:
    - This function requires the following libraries: tiff, box, np, plt from matplotlib.

    Example usage:
    ```
    sg_obj = SGobject(...)
    identifier = 'object123'
    image_path = '/path/to/image.tif'
    fig = show_neighberhood(sg_obj, identifier, image_path)
    plt.show()
    ```
    """
    # Load the TIFF file
    image = tiff.imread(image_path)

    bounds_tuple = tuple(map(int, bounds_tuple))

    # unpack the bounds from the passsed tuple
    (range_x_lower, range_y_lower, range_x_upper, range_y_upper) =  bounds_tuple

    # Ensure the ranges are within image bounds
    range_y_lower = max(range_y_lower, 0)
    range_y_upper = min(range_y_upper, image.shape[1])
    range_x_lower = max(range_x_lower, 0)
    range_x_upper = min(range_x_upper, image.shape[2])

    print(range_y_lower,range_y_upper, range_x_lower,range_x_upper)

    # Create the boolean mask
    # spot_in_range = ((spot_coords[:, 1] > range_y_lower) & (spot_coords[:, 1] < range_y_upper)) & \
    #                 ((spot_coords[:, 0] > range_x_lower) & (spot_coords[:, 0] < range_x_upper))
    # print(np.shape(spot_in_range))

    # Crop the fourth channel using the bounding rectangle
    cropped_image = image[3, range_y_lower:range_y_upper, range_x_lower:range_x_upper]

    # Check if cropped_image is non-empty
    if cropped_image.size == 0:
        print(f"No pixels found in the specified range for coordinates {coordinates}")
        return

    gray_array = np.array(cropped_image)

    # Normalize the pixel values to the range 0-255
    normalized_gray = (gray_array - gray_array.min()) / (gray_array.max() - gray_array.min()) * 255
    normalized_gray = normalized_gray.astype(np.uint8)

    # Increase the saturation of the blue channel
    saturated_blue = normalized_gray * 1.5  # Increase intensity by 50%
    saturated_blue = np.clip(saturated_blue, 0, 255).astype(np.uint8)  # Ensure values are within 0-255

    # Create a new image with the blue channel based on the saturated grayscale values
    blue_image_array = np.zeros((gray_array.shape[0], gray_array.shape[1], 3), dtype=np.uint8)
    # blue_image_array[:, :, 2] = saturated_blue  # Set the blue channel
    blue_image_array[:, :, 2] = normalized_gray  # Set the blue channel

    # Create a figure
    # fig, ax = plt.subplots()
    ax.imshow(blue_image_array,extent=[range_x_lower,range_x_upper,range_y_lower,range_y_upper],origin='lower')


    # ax.axis('off')  # Hide axis

    return ax



def plot_polygons_and_points(sg_obj, identifiers, id_field='object_id', 
                             gene_names=None,annotate=True,image_scale=1.0,
                                interior_marker='o',exterior_marker='x',
                                focal_outline_color='red',other_outline_color='black',
                                interior_edgecolor=None,central_polygon_ix=0,single_mode=True,
                                marker_size=50,lw=1,color_map=None,label=None,ax=None,
                                show_image=False,image_path=None,
                                annotate_cells=False,annotation_color='w',
                                annotation_fontsize=8
                                ,**kwargs):
        if sg_obj.gdf is None or sg_obj.assigned_points_gdf is None:
            print("Error: Ensure both gdf and assigned_points_gdf are loaded.")
            return
        
        # make kwargs back into dictionary
        
    

        polygon_gdf = sg_obj.gdf[sg_obj.gdf[id_field].isin(identifiers)]

        if polygon_gdf.empty:
            print(f"No polygon found with {id_field} == {identifier}")
            return

        if single_mode:
            first_polygon_geometry = polygon_gdf.geometry.iloc[central_polygon_ix]
            minx, miny, maxx, maxy = first_polygon_geometry.geometry.bounds
        else:
            all_polygon_bounds = polygon_gdf.geometry.bounds
            minx = min(all_polygon_bounds.minx)
            miny = min(all_polygon_bounds.miny)
            maxx = max(all_polygon_bounds.maxx)
            maxy = max(all_polygon_bounds.maxy)

        dx = (maxx - minx) * 0.5 * image_scale
        dy = (maxy - miny) * 0.5 * image_scale
        expanded_bbox = box(minx - dx, miny - dy, maxx + dx, maxy + dy)

        other_polygons = sg_obj.gdf[sg_obj.gdf.geometry.intersects(expanded_bbox) & (~sg_obj.gdf[id_field].isin(identifiers))]
        
        
        if gene_names is not None:
            if isinstance(gene_names, str):
                gene_names = [gene_names]
            # first pick the subset of points corresponding to the gene names
            points_within_bbox = sg_obj.assigned_points_gdf[sg_obj.assigned_points_gdf['name'].isin(gene_names)]
            # then filter to those within the expanded bbox
            points_within_bbox = points_within_bbox[points_within_bbox.geometry.within(expanded_bbox)]
        else:
            # if no gene names are passed, get all the ones in the expanded bbox
            points_within_bbox = sg_obj.assigned_points_gdf[sg_obj.assigned_points_gdf.geometry.within(expanded_bbox)]

        if ax is None:
            fig, ax = plt.subplots()

        if show_image:
            if image_path == None:
                print('No image path found')
                return
            
            bounds_tuple = (minx - dx, miny - dy, maxx + dx, maxy + dy)

            ax = show_neighborhood_subimage(image_path,bounds_tuple,ax=ax)



        polygon_gdf.boundary.plot(ax=ax, color=focal_outline_color, linewidth=lw)
        other_polygons.boundary.plot(ax=ax, color=other_outline_color, linewidth=lw)

        # Generate a unique color for each name
        unique_names = points_within_bbox['name'].unique()
        # sort by the order of the gene names if it exists
        if gene_names is not None:
            unique_names = sorted(unique_names, key=lambda x: gene_names.index(x))

        if color_map is None:
            color_map = {name: plt.cm.tab20(i % 20) for i, name in enumerate(unique_names)}
        # else:


        # Plot points, label them, and use consistent colors for names
        for name, group in points_within_bbox.groupby('name'):
            interior_points = group[group[id_field].isin(identifiers)]
            exterior_points = group[~group[id_field].isin(identifiers)]
            
            # Plot interior points with 'o' marker style
            if not interior_points.empty:
                ax.scatter(interior_points.geometry.x, interior_points.geometry.y,
                           marker=interior_marker, s=marker_size, edgecolor=interior_edgecolor, 
                           color=color_map[name],lw=0,**kwargs)

            # Plot exterior points with 'x' marker style
            if not exterior_points.empty:
                ax.scatter(exterior_points.geometry.x, exterior_points.geometry.y, marker=exterior_marker, 
                           s=marker_size, color=color_map[name],lw=0,**kwargs)
            
            if annotate:
                # Labeling remains the same for all points
                for x, y in zip(group.geometry.x, group.geometry.y):
                    ax.text(x, y, name, fontsize=8, ha='right')

        # add names for the cells
        if annotate_cells:
            for x,y,name in zip(polygon_gdf.geometry.centroid.x,polygon_gdf.geometry.centroid.y,polygon_gdf['object_id'].values):
                ax.text(x, y, name, fontsize=annotation_fontsize, ha='center',color=annotation_color)
            for x,y,name in zip(other_polygons.geometry.centroid.x,other_polygons.geometry.centroid.y,other_polygons['object_id'].values):
                ax.text(x, y, name, fontsize=annotation_fontsize, ha='center',color=annotation_color)


        ax.set_xlim([minx - dx, maxx + dx])
        ax.set_ylim([miny - dy, maxy + dy])
        ax.set_title(f"Polygons {label} and Surrounding Area")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        return ax

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering as AggCluster

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)