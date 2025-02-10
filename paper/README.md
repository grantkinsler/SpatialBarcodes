This directory contains the necessary code for running analysis and plotting figures for the SpaceBar paper.

extractionScripts contains scripts to convert raw data to extracted data as well as some minimal analysis.
The scripts should be run in order:

**1_process_to_SGobject_and_callbarcodes.py** assigns spots to cells and calls barcode and clone identity for cells. 

**2_erode_tumor.py** assigns cells to spatial rings from the edge of tumor.

**3_calculuate_clone_scores.ipynb** calculates clone scores for all genes.

**4_calculuate_space_scores.ipynb** calculates space scores for all genes.


plotScripts contains scripts that do the bulk of the plotting and analysis of the processed data.

**calculate_error_rate.ipynb** calculates the fraction of cells that received the same barcode combination ("false clone rate") in the in vitro data.

**calculate_moi_cells_in_dish.ipynb** calculates the MOI of the barcodes in the experiment.

**in_vitro_plotting.ipynb** plots examples of clones and gene expression.

**in_vitro_sister_similarity.ipynb** analysis on the similarity of gene expression of clones in vitro.

**in_vivo_plotting.ipynb** plots examples of detection in vivo.

**plot_barcode_overlap_comparison.ipynb** plots the fraction of non-sister neighbor cells that share barcodes ("mis-assignment rate").

**plot_clone_space_examples.ipynb** plots examples of the clone and spaces scores in the tumor.

**plot_clone_space_scores.ipynb** plots comparision of clone and space scores.

**rate of neighbors barcode overlap clustered all roi single cell algo.ipynb** calculates the fraction of neighboring cells that have overlapping barcodes for the mis-assignment rate with clones assigned by barcode clustering.

**rate of neighbors barcode overlap unclustered all roi single cell algo.ipynb** calculates the fraction of neighboring cells that have overlapping barcodes for the mis-assignment rate with clones assigned by simple barcode thresholding