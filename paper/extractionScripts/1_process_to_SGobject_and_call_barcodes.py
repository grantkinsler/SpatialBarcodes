import numpy as np
# import cv2
import tifffile as tiff
from shapely.geometry import Polygon, mapping, shape, box
from skimage import io
from skimage.measure import label, regionprops
# import matplotlib.pyplot as plt
from rasterio import features
import scanpy as sc
import pandas as pd
import geopandas as gpd

import pickle
import scipy

from SGanalysis.SGobject import SGobject 


from sklearn.cluster import AgglomerativeClustering as AggCluster

import sys
import os

tools_path = '../helperScripts/tools.py'

sys.path.append(os.path.dirname(os.path.expanduser(tools_path)))
import tools


partially_processed = False
cutoff = 3
barcode_cols = ['bc_{:03d}'.format(i) for i in range(1,97)]

from SGanalysis.SGobject import SGobject

for roi_name,files in tools.roi_file_paths.items():

    print(f'Processing {roi_name}...')

    segmentation_file = files['segmentation_file']
    spots_file = files['spots_file']
    out_path = files['out_path']
    
    if not partially_processed:
        # Create an instance of SGobject
        sg_obj = SGobject()

        # Convert a TIFF image to polygons and store them in a GeoDataFrame
        print("Running mask_to_objects...")
        sg_obj.mask_to_objects(segmentation_file)

        sg_obj.load_points(spots_file)

        sg_obj.dilate_objects(10)

        ## associate spots with segmentation

        sg_obj.create_cell_gene_table()

        # with open(f'{out_path}/sg_object_dilate10_20240718.pkl', 'wb') as f:
        #     pickle.dump(sg_obj, f)
            
        matrix = sg_obj.get_cell_gene_table_df()

        if matrix.index.name == 'object_id':
            matrix = matrix.reset_index()
        matrix['object_id'] = [str(int(x)) for x in matrix['object_id']]
        matrix.set_index('object_id',inplace=True)

        sg_obj.gdf['object_id'] = [str(int(x)) for x in sg_obj.gdf['object_id']]
        sg_obj.gdf.set_index('object_id',inplace=True)

        sg_obj.gdf['nucleus_centroid'] = sg_obj.gdf['nucleus'].centroid.values
        sg_obj.gdf['center_x'] = sg_obj.gdf['nucleus'].centroid.x.values
        sg_obj.gdf['center_y'] = sg_obj.gdf['nucleus'].centroid.y.values
        sg_obj.gdf['area'] = sg_obj.gdf['nucleus_dilated'].area.values

        sg_obj.gdf

        barcode_cols = ['bc_{:03d}'.format(i) for i in range(1,97)]

        with open(f'{out_path}/sg_object_dilate10_20240718.pkl', 'wb') as f:
            pickle.dump(sg_obj, f)


        ## traditional barcode calling (using cutoff of 3)
        cell_barcodes = {}
        # matrix.set_index('object_id',   inplace=True)
        # df.set_index('object_id')

        df = matrix
        df['cell_id'] = df.index

        for cell_id in df['cell_id']:
            this_cell = df[df['cell_id']==cell_id]
            cell_barcodes[cell_id] = []

            for bc in barcode_cols:
                if this_cell[bc].values[0] >= cutoff:
                    cell_barcodes[cell_id].append(bc)

        df['called_barcodes'] = cell_barcodes.values()
        df['n_called_barcodes'] = [len(bc_set) for bc_set in cell_barcodes.values()]
        df['barcode_names'] = ['-'.join(sorted(bc_set)) for bc_set in cell_barcodes.values()]

        df = pd.merge(df,sg_obj.gdf,how='left',left_index=True,right_index=True)

        df.to_csv(f'{out_path}/cell_by_gene_matrix_dilate10_20240718_withbarcodes_atleast{cutoff}.csv')
    else:
#    	with open(f'{out_path}/sg_object_dilate10_20240718.pkl', 'rb') as f:
#		sg_obj = pickle.load(f)
		
	    df = pd.read_csv(f'{out_path}/cell_by_gene_matrix_dilate10_20240718_withbarcodes_atleast{cutoff}.csv')


    total_bc_threshold = 10

    # include cells with at least 10 barcode spots
    has_bcs = df[df[barcode_cols].sum(axis=1) >= 10]
    matrix = has_bcs[barcode_cols]

    print('Clustering barcodes...')
    print('Calculating distance matrix...')

    matrix_norm = matrix.div(matrix.sum(axis=1), axis=0)
    braycurtis_dist = scipy.spatial.distance.pdist(matrix_norm,metric='braycurtis')
    
    print('Performing clustering...')

    # threshold = 0.4
    threshold = 0.2
    # cluster = AggCluster(distance_threshold=threshold,n_clusters=None,linkage='average',metric='precomputed').fit(scipy.spatial.distance.squareform(braycurtis_dist))
    cluster = AggCluster(distance_threshold=threshold,n_clusters=None,linkage='average',affinity='precomputed').fit(scipy.spatial.distance.squareform(braycurtis_dist))


    gene_cols = [col for col in df.columns if 'bc_' not in col and col not in ['cell_id','called_barcodes','n_called_barcodes','barcode_names','area','center_x','center_y','nucleus','nucleus_centroid','nucleus_dilated']]    

    adata_genes = sc.AnnData(df[gene_cols])


    cols= ['cell_id','n_called_barcodes','barcode_names','called_barcodes','area','center_x','center_y']
    # for col in :
    #     adata_genes.obs[col] = df[col]

    adata_genes.obs = df[cols]


    cluster_name_cutoff = 3

    obj_clusters = {obj_id:clus for obj_id,clus in zip(matrix.index,cluster.labels_)}

    cluster_objects = {}
    cluster_barcode_names = {}
    cluster_found_barcodes = {}
    cluster_n_found_barcodes = {}

    for clu in np.unique(cluster.labels_):
        cluster_objects[clu] = matrix.index[cluster.labels_ == clu]

        avg_bc_counts = matrix[cluster.labels_ == clu].median(axis=0)
        # print(avg_bc_counts)
        # break

        found_bcs = []

        for bc in barcode_cols:
            if avg_bc_counts[bc] > cutoff:
                found_bcs.append(bc)

#        print(clu,found_bcs)

        cluster_found_barcodes[clu] = found_bcs
        cluster_n_found_barcodes[clu] = len(found_bcs)

        cluster_barcode_names[clu] = '-'.join(sorted(found_bcs))

    adata_genes.obs['bc_cluster'] = [obj_clusters[obj_id] if obj_id in obj_clusters else np.nan for obj_id in adata_genes.obs.index]
    adata_genes.obs['bc_cluster_n_bcs'] = [cluster_n_found_barcodes[obj_clusters[obj_id]] if obj_id in obj_clusters else np.nan for obj_id in adata_genes.obs.index]
    adata_genes.obs['bc_cluster_found_bcs'] = [cluster_found_barcodes[obj_clusters[obj_id]] if obj_id in obj_clusters else np.nan for obj_id in adata_genes.obs.index]
    adata_genes.obs['bc_cluster_bc_names'] = [cluster_barcode_names[obj_clusters[obj_id]] if obj_id in obj_clusters else np.nan for obj_id in adata_genes.obs.index]

    adata_genes.obs.to_csv(f'{out_path}/cell_by_gene_matrix_dilate10_20240718_withbarcodes_clustering_{total_bc_threshold}bcs_{threshold}thresh.csv')
