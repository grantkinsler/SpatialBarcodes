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

from matplotlib import colors

import pickle
import scipy

import seaborn as sns

from SGanalysis.SGobject import SGobject

import matplotlib
# %matplotlib inline
# matplotlib.use('Qt5Agg')
# %matplotlib qt


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

def get_polygon_barcodes(df,identifier,id_field='cell_id'):

    if len(df[df[id_field]==identifier]['called_barcodes'].values) > 0:

        return df[df[id_field]==identifier]['called_barcodes'].values[0]

    else:
        return []

def get_all_barcodes_in_region(df,identifiers,id_field='cell_id'):

    bc_list = []

    for identifier in identifiers:
        bc_list += get_polygon_barcodes(df,identifier,id_field=id_field)

    return bc_list







def show_neighberhood(sg_obj, identifier, image_path, id_field='object_id', image_scale=1.0):
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



def plot_polygon_and_points(sg_obj, identifier, id_field='object_id', gene_names=None,annotate=True,image_scale=1.0,
                                interior_marker='o',exterior_marker='x',marker_size=50,color_map=None):
        if sg_obj.gdf is None or sg_obj.assigned_points_gdf is None:
            print("Error: Ensure both gdf and assigned_points_gdf are loaded.")
            return

        polygon_gdf = sg_obj.gdf[sg_obj.gdf[id_field] == identifier]

        if polygon_gdf.empty:
            print(f"No polygon found with {id_field} == {identifier}")
            return

        first_polygon_geometry = polygon_gdf.geometry.iloc[0]

        minx, miny, maxx, maxy = first_polygon_geometry.bounds
        dx = (maxx - minx) * 0.5 * image_scale
        dy = (maxy - miny) * 0.5 * image_scale
        expanded_bbox = box(minx - dx, miny - dy, maxx + dx, maxy + dy)

        other_polygons = sg_obj.gdf[sg_obj.gdf.geometry.intersects(expanded_bbox) & (sg_obj.gdf[id_field] != identifier)]
        
        
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

        fig, ax = plt.subplots()
        polygon_gdf.boundary.plot(ax=ax, color='red', linewidth=2)
        other_polygons.boundary.plot(ax=ax, color='black', linewidth=1)

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
            interior_points = group[group[id_field] == identifier]
            exterior_points = group[group[id_field] != identifier]
            
            # Plot interior points with 'o' marker style
            if not interior_points.empty:
                ax.scatter(interior_points.geometry.x, interior_points.geometry.y, marker=interior_marker, s=marker_size, edgecolor='black', color=color_map[name])
            
            # Plot exterior points with 'x' marker style
            if not exterior_points.empty:
                ax.scatter(exterior_points.geometry.x, exterior_points.geometry.y, marker=exterior_marker, s=marker_size, color=color_map[name])
            
            if annotate:
                # Labeling remains the same for all points
                for x, y in zip(group.geometry.x, group.geometry.y):
                    ax.text(x, y, name, fontsize=8, ha='right')

        ax.set_xlim([minx - dx, maxx + dx])
        ax.set_ylim([miny - dy, maxy + dy])
        ax.set_title(f"Polygon {identifier} and Surrounding Area")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.show()