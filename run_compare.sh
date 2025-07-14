#!/bin/bash

TEST_PDF=demo/samples_tables.pdf

# Initialize conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate marker162

marker_single $TEST_PDF --output_format json --output_dir output/marker162


CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate minerumarker


magic-pdf -p $TEST_PDF  -o output/mineru1310_markertables -m auto

magic-pdf -p $TEST_PDF  -o output/mineru1310_rapidtables -m auto

