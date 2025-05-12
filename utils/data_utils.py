import pickle
import numpy as np
import rasterio
from datetime import datetime

def load_data(filepath):
    with open(filepath, 'rb') as f:
        print(f"Loading data from {filepath}...")
        return pickle.load(f)

def convert_tr_dt_to_dt(date_tensor):
    return [datetime.fromtimestamp(ts.item()) for ts in date_tensor]

def get_projection_string(prj_file_path):
    with open(prj_file_path, 'r') as file:
        projection_string = file.read()
    return projection_string

# Function to read LULC data
def process_lulc_data(filepath):
    with rasterio.open(filepath) as src:
        lulc_data = src.read(1)  # Read the first band
    unique_classes, counts = np.unique(lulc_data, return_counts=True)
    valid_classes = unique_classes[(unique_classes != 0) & (unique_classes != 255)]
    class_to_index = {class_val: i + 1 for i, class_val in enumerate(valid_classes)}
    class_to_index[0] = 0
    class_to_index[255] = 0
    class_to_index[256] = 0
    return lulc_data, class_to_index

# Function to apply the mapping
def remap_lulc_data(lulc_data, mapping):
    remapped_data = np.copy(lulc_data)
    for class_value, mapped_value in mapping.items():
        remapped_data[remapped_data == class_value] = mapped_value
    return remapped_data
