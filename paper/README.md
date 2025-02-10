This directory contains the necessary code for running analysis and plotting figures for the SpaceBar paper.

extractionScripts contains scripts to convert raw data to extracted data as well as some minimal analysis.
The scripts should be run in order:

**1_process_to_SGobject_and_callbarcodes.py** assigns spots to cells and calls barcode and clone identity for cells. 

**2_erode_tumor.py** assigns cells to spatial rings from the edge of tumor.

**3_calculuate_clone_scores.ipynb** calculates clone scores for all genes.

**4_calculuate_space_scores.ipynb** calculates space scores for all genes.


plotScripts contains scripts that do the bulk of the plotting and analysis of the processed data.
