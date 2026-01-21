#!/usr/bin/env bash

# Optional: set OUT_DIR to control where outputs are written.
# Defaults to the sibling "data" folder if not provided.
OUT_DIR=${OUT_DIR:-../data}

# IU - xray
python preprocess_mask.py --dataset_name iu_xray --data_path "$OUT_DIR"

# mimic - cxr
# python preprocess_mask.py \
#   --dataset_name mimic_cxr \
#   --data_path "$OUT_DIR"
