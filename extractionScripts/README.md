This directiory contains all of the scripts to process data for the SpaceBar paper. In conjunction with the plotting scripts available in ../plotScripts, you can recreate all of the analysis and figure generation for the paper.

The scripts perform the following tasks:

1_process_to_SGobject_and_call_barcodes.py - This script takes data from the SGAnalysis software (called transcripts and nuclear information), assigns transcritps to cells, calls barcodes per cell, and performs clustering on the barcodes to call clones. 

2_erode_tumor.ipynb - This script takes a pre-defined (or user-input) mask for a tumor and erodes the edges into spatial groupings. We use this to calculate the space scores.

3_calculate_clone_scores.ipynb - Calculates clone scores to identify genes that have clonal expression patterns.

4_calculate_space_scores.ipynb - Calculates space scores to identify genes with spatial expression patterns.

5_extraction_region_information.ipynb - Calculates the size of regions of interest from the imaging.

6_invitro_expression_helper.ipynb - Calculates the mean expression of genes and barcodes from the all in-vitro ROIs.

7_invivo_expression_helper.ipynb - Calculates the mean expression of genes and barcodes from all the in-vivo ROIs.

8_neighbor_barcode_overlap_unclustered.ipynb - Calcualtes the fraction of non-sister neighbors that share a substantial number of barcodes, for unclustered data.

8_neighbor_barcode_overlap_clustered.ipynb - Same as (8) but for clustered data