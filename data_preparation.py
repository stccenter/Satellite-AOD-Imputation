import numpy as np
import numpy.ma as ma

import h5py
import pickle

from datetime import datetime, timedelta
import time, os 
import gc

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

abbr_dict_region = {'Northeast': 'NE',
             'NorthernRockies': 'NR',
             'Northwest': 'NW',
             'OhioValley': 'OV',
             'South': 'S',
             'Southeast': 'SE',
             'Southwest':'SW',
             'UpperMidwest': 'UM',
             'West': 'W',
}

aod_fv = -9999
fill_value = np.nan
is_subset = False
koi = ['AODANA', 'BC', 'BLH', 'DU', 'ELEVATION', 'MAIAC', 'R', 'T2M', 'TCC', 'WIND_SPEED']

region_name = 'Northeast'
region_abbr = abbr_dict_region.get(region_name)

base_dir = f'./data/'

hdf5_2021 = f"{base_dir}/HDF5/{region_name}_20210101_20211231_20241126.hdf5"
hdf5_2022 = f"{base_dir}/HDF5/{region_name}_20220101_20221231_20241126.hdf5"
hdf5_2023 = f"{base_dir}/HDF5/{region_name}_20230101_20231231_20241126.hdf5"

def scale_data_with_nans(data, scalers):
    channels = data.shape[-1]
    scaled_data = np.full(data.shape, np.nan, dtype=np.float32)
    
    for i in range(channels):
        print(f'Scaling channel {i}')
        channel_data = data[:, :, :, i]
        # Flatten the array and filter out NaN values for scaling
        valid_data = channel_data[~np.isnan(channel_data)].reshape(-1, 1)
        
        # Transform valid data using the scaler fitted on training data
        if valid_data.size > 0:
            scaled_valid_data = scalers[i].transform(valid_data).flatten()
            
            # Assign the scaled values back to the appropriate positions in the scaled array
            valid_indices = ~np.isnan(channel_data)
            scaled_data[:, :, :, i][valid_indices] = scaled_valid_data

    return scaled_data

def scale_y(y_train, y_val, y_test):
     # Apply binary mask to y_data, setting masked areas to NaN
    y_train_masked = np.where((y_train == -9999) | (y_train == -1), np.nan, y_train)
    # Initialize a scaler for y_data
    y_scaler = StandardScaler()
    # Flatten y_data, excluding NaN values
    valid_y_train = y_train_masked[~np.isnan(y_train_masked)].reshape(-1, 1)
    # Fit and transform the valid y_data
    scaled_valid_y_train = y_scaler.fit_transform(valid_y_train)
    # Create a full-sized array filled with NaNs to hold the scaled y_data
    scaled_y_train = np.full(y_train.shape, np.nan)
    # Place the scaled data back into the scaled_y_data array at the valid positions
    scaled_y_train[~np.isnan(y_train_masked)] = scaled_valid_y_train.flatten()
    # y_data_scaled now contains the scaled target variable, with areas outside the ROI preserved as NaN
    y_train_scaled = scaled_y_train
    
    y_val_masked = np.where((y_val == -9999) | (y_val == -1), np.nan, y_val)
    y_test_masked = np.where((y_test == -9999) | (y_test == -1), np.nan, y_test)
    
    valid_val_indices = ~np.isnan(y_val_masked)
    y_val_flat = y_val_masked[valid_val_indices].reshape(-1, 1)
    y_val_scaled = np.full(y_val_masked.shape, np.nan)
    y_val_scaled[valid_val_indices] = y_scaler.transform(y_val_flat).flatten()
    
    # Flatten, scale, and reshape back the test target data
    valid_test_indices = ~np.isnan(y_test_masked)
    y_test_flat = y_test_masked[valid_test_indices].reshape(-1, 1)
    y_test_scaled = np.full(y_test_masked.shape, np.nan)
    y_test_scaled[valid_test_indices] = y_scaler.transform(y_test_flat).flatten()
    return y_train_scaled, y_val_scaled, y_test_scaled, y_scaler

def custom_split(x_data, y_data, miss_matrix, date_arr, month_arr, season_arr):
    # Step 1: Splitting into train_val (80%) and test (20%)
    split_idx_test = int(len(x_data) * 0.8)
    
    x_train_val = x_data[:split_idx_test]
    x_test = x_data[split_idx_test:]
    
    y_train_val = y_data[:split_idx_test]
    y_test = y_data[split_idx_test:]
    
    miss_matrix_train_val = miss_matrix[:split_idx_test]
    miss_matrix_test = miss_matrix[split_idx_test:]
    
    dates_train_val = date_arr[:split_idx_test]
    dates_test = date_arr[split_idx_test:]
    
    month_train_val = month_arr[:split_idx_test]
    month_test = month_arr[split_idx_test:]
    
    season_train_val = season_arr[:split_idx_test]
    season_test = season_arr[split_idx_test:]
    
    # Step 2: Splitting train_val into train (85% of train_val) and val (15% of train_val)
    split_idx_val = int(len(x_train_val) * 0.85)
    
    x_train = x_train_val[:split_idx_val]
    x_val = x_train_val[split_idx_val:]
    
    y_train = y_train_val[:split_idx_val]
    y_val = y_train_val[split_idx_val:]
    
    miss_matrix_train = miss_matrix_train_val[:split_idx_val]
    miss_matrix_val = miss_matrix_train_val[split_idx_val:]
    
    dates_train = dates_train_val[:split_idx_val]
    dates_val = dates_train_val[split_idx_val:]
    
    month_train = month_train_val[:split_idx_val]
    month_val = month_train_val[split_idx_val:]
    
    season_train = season_train_val[:split_idx_val]
    season_val = season_train_val[split_idx_val:]
    return x_train, x_val, x_test, y_train, y_val, y_test, miss_matrix_train, miss_matrix_val, miss_matrix_test, dates_train, dates_val, dates_test, month_train, month_val, month_test, season_train, season_val, season_test
    
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'fall'

def iso_to_gregorian(iso_year, iso_week, iso_day):
    """
    Convert ISO year, week, and day into a Gregorian (standard) date.
    """
    fourth_jan = datetime(iso_year, 1, 4)
    delta_days = iso_day - fourth_jan.isoweekday()
    year_start = fourth_jan + timedelta(days=delta_days)
    return year_start + timedelta(weeks=iso_week-1)

def process_hdf5_file(hdf_path, koi, is_subset, aod_fv, fill_value):
    datasets = {}
    with h5py.File(hdf_path, 'r') as file:
        print(f'Read the lat and long points for file {hdf_path}')
        latitude = file['latitude'][:]
        longitude = file['longitude'][:]

        print('squaring the coordinates')
        lat_square = np.square(latitude)
        lon_square = np.square(longitude)

        print('finding the product of lat and long')
        lat_lon_product = np.multiply(latitude, longitude)

        datasets = {'latitude': latitude, 'longitude': longitude, 'lat_lon_product': lat_lon_product}

        time_data = [t.decode("utf-8") if isinstance(t, bytes) else t for t in file['time'][:]]
        time_datetimes = [datetime.strptime(t, '%Y%m%d') for t in time_data]

        datasets['time'] = time_data  # Already in string format

        # Calculate week numbers; this works for both subset and full dataset cases
        weeks = [t.isocalendar().week for t in time_datetimes]

        for name in file.keys():
            if name in koi:
                print(f'Processing variable: {name}')

                dataset = file[name][:]

                if name == 'AODANA':
                    data_bm = np.where(np.isnan(dataset), 0, 1)

                if name == 'MAIAC':
                    if dataset[0,0,0] == -9999:
                        print('-9999 exists in the array as first pixel')
                        dataset = np.where(dataset == -9999, np.nan, dataset)
                    # dataset = np.where(dataset == aod_fv, -1, dataset)
                    dataset = np.where(np.isnan(dataset), -1, dataset)
                    dataset = np.where(data_bm == 0, aod_fv, dataset)
                    data_mask = ma.masked_where(dataset == aod_fv, dataset)
                else:
                    dataset = np.where(data_bm == 0, fill_value, dataset)
                datasets[name] = dataset

    return datasets, data_bm

def get_data(hdf5_2021, hdf5_2022, koi, is_subset, aod_fv, fill_value):
    process_start_time = time.time()
    # Process each file
    print('1. Create dataset dictionary.')
    datasets_2021, data_bm_2021 = process_hdf5_file(hdf5_2021, koi, is_subset, aod_fv, fill_value)
    datasets_2022, data_bm_2022 = process_hdf5_file(hdf5_2022, koi, is_subset, aod_fv, fill_value)

    # Merge the datasets
    spatial_keys = ['latitude', 'longitude', 'lat_lon_product']
    time_key = 'time'

    # Initialize empty dictionaries for the merged datasets and spatial data
    datasets = {}
    spatial_data = {}
    
    # Loop through keys in datasets_2021
    for key in datasets_2021:
        if key in datasets_2022:
            if key not in spatial_keys and key != time_key:
                # Concatenate data for keys present in both datasets, excluding spatial and time keys
                print(f'Key exists and concatenated: {key}')
                datasets[key] = np.concatenate((datasets_2021[key], datasets_2022[key]), axis=0)
            elif key in spatial_keys:
                # Directly assign spatial data from datasets_2021 to the spatial_data dictionary
                # Assuming you want to keep spatial data from 2021 or handle it differently
                spatial_data[key] = datasets_2021[key]

    data_bm = np.concatenate((data_bm_2021, data_bm_2022), axis=0)

    # Free memory from individual year datasets
    del data_bm_2021, data_bm_2022
    gc.collect()  # Explicitly clear the garbage

    latitude = spatial_data['latitude']
    longitude = spatial_data['longitude']

    print('2. Split the data into x and y.')
    time_steps = datasets['AODANA'].shape[0]

    expanded_spatial_data = []
    for key in spatial_data.keys():
        expanded_data = np.repeat(spatial_data[key][np.newaxis, :, :], time_steps, axis=0)
        print(f'spatial key {key} has shape {expanded_data.shape}')
        masked_expanded_data = np.where(data_bm == 0, np.nan, expanded_data)
        expanded_spatial_data.append(masked_expanded_data)

    # Free memory from spatial data dictionary
    del spatial_data
    gc.collect()
    
    # Prepare feature data identified by feature_keys for stacking
    feature_keys = [key for key in datasets.keys() if key not in spatial_keys + ['MAIAC'] + ['time']]
    feature_data = [datasets[key] for key in feature_keys]
    
    # Stacking both feature data and expanded spatial data
    cv_data = np.stack(feature_data + expanded_spatial_data, axis=-1)
    y_data = datasets['MAIAC']  # Target variable

    # Free memory from merged datasets
    del datasets
    gc.collect()

    print('3. Create the miss matrix')
    # Initialize the missing matrix with zeros
    miss_matrix = np.zeros_like(y_data, dtype=int)
    # Mark observed values as 1 (Observed AOD values are greater than 0 and not equal to -9999)
    observed_mask = (y_data != -1) & (y_data != -9999)
    miss_matrix[observed_mask] = 1

    # Free memory from intermediate mask variables
    del observed_mask
    gc.collect()

    print('4. Get the season and month arrays')
    # Define the start date (January 1, 1990)
    time_data = np.concatenate((datasets_2021['time'], datasets_2022['time']), axis=0)
    date_arr = np.array([datetime.strptime(date,'%Y%m%d') for date in time_data])

    # Free memory from raw time data
    del time_data
    gc.collect()

    # Extract month and season from the date array
    month_arr = np.array([date.month for date in date_arr])
    seasons = np.array([get_season(month) for month in month_arr])

    # For seasons
    # Map each season to an integer
    season_mapping = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
    season_arr = np.array([season_mapping[season] for season in seasons])

    x_data = cv_data

    # Free memory from raw arrays
    del cv_data, seasons
    gc.collect()

    num_samples, height, width, channels = x_data.shape
    print('7. Split - train test and validation')
    x_train, x_val, x_test, y_train, y_val, y_test, miss_matrix_y_train, miss_matrix_y_val, miss_matrix_y_test, d_train, d_val, d_test, month_train, month_val, month_test, season_train, season_val, season_test = custom_split(x_data, y_data, miss_matrix, date_arr, month_arr, season_arr)

    # Free memory from unused arrays
    del x_data, y_data, miss_matrix, date_arr, month_arr, season_arr
    gc.collect()
    
    print('8. Perform scaling of x - train test and validation')
    scalers = [StandardScaler() for _ in range(x_train.shape[-1])]
    
    # Adjust binary_mask to match x_train shape for broadcasting
    binary_mask_expanded = np.broadcast_to(data_bm[0][np.newaxis, :, :, np.newaxis], x_train.shape).astype(np.float32)
    
    # Initialize the scaled array with NaNs
    x_train_scaled = np.full(x_train.shape, np.nan, dtype=np.float32)
    
    # Scaling logic for each channel
    for i in range(x_train.shape[-1]):
        print(f'Processing channel {i}')
        channel_data = x_train[:, :, :, i]
        
        # Mask with NaN where binary mask is 0
        channel_data_masked = np.where(binary_mask_expanded[:, :, :, i] == 0, np.nan, channel_data)
        
        # Flatten the array and filter out NaN values for scaling
        valid_data = channel_data_masked[~np.isnan(channel_data_masked)].reshape(-1, 1)
        
        # Fit and transform valid data using the scaler
        if valid_data.size > 0:
            scaler = scalers[i]
            scaled_valid_data = scaler.fit_transform(valid_data).flatten()
            
            # Assign the scaled values back to the appropriate positions in the scaled array
            valid_indices = ~np.isnan(channel_data_masked)
            x_train_scaled[:, :, :, i][valid_indices] = scaled_valid_data

        # Store the scaler for inverse transformation later, if needed
        scalers[i] = scaler
    
    # Free memory from unscaled train data
    del x_train
    gc.collect()
    
    print('9. Scale the x test and validation with y-train scaler')
    # Usage example for validation and test data:
    x_val_scaled = scale_data_with_nans(x_val, scalers)
    x_test_scaled = scale_data_with_nans(x_test, scalers)

    # Free memory from unscaled validation and test data
    del x_val, x_test
    gc.collect()
    
    print('10. Perform scaling of y - train test and validation')
   
    # y_train, y_val, y_test, y_scaler = scale_y(y_train, y_val, y_test)
    y_train_model = np.expand_dims(y_train, axis=1)
    y_test_model = np.expand_dims(y_test, axis=1)
    y_val_model = np.expand_dims(y_val, axis=1)
    
    miss_y_train_model = np.expand_dims(miss_matrix_y_train, axis=1)
    miss_y_test_model = np.expand_dims(miss_matrix_y_test, axis=1)
    miss_y_val_model = np.expand_dims(miss_matrix_y_val, axis=1)

    # Free memory from unscaled target data
    del y_train, y_val, y_test
    gc.collect()

    x_train_model = x_train_scaled.transpose(0, 3,  1, 2)  
    x_test_model = x_test_scaled.transpose(0, 3,  1, 2)  
    x_val_model = x_val_scaled.transpose(0, 3,  1, 2)  

    process_end_time = time.time()
    # Calculate the execution time for the current iteration and convert to minutes
    process_time_minutes = (process_end_time - process_start_time) / 60
    print(f'Execution time for pre-processing: {process_time_minutes:.2f} minutes')
    return y_train_model, y_test_model, y_val_model, miss_y_train_model, miss_y_test_model, miss_y_val_model, x_train_model, x_test_model, x_val_model, d_train, d_test, d_val, month_train, month_test, month_val, season_train, season_test, season_val, data_bm, scalers, latitude, longitude

def get_data_inference(hdf5_infer, koi, is_subset, aod_fv, fill_value):
    process_start_time = time.time()
    # Process each file
    print('1. Create dataset dictionary.')
    datasets_infer, data_bm = process_hdf5_file(hdf5_infer, koi, is_subset, aod_fv, fill_value)

    # Merge the datasets
    spatial_keys = ['latitude', 'longitude', 'lon_square', 'lat_square', 'lat_lon_product']
    time_key = 'time'

    # Initialize empty dictionaries for the merged datasets and spatial data
    datasets = {}
    spatial_data = {}
    
    for key in datasets_infer:
        if key not in spatial_keys and key != time_key:
            datasets[key] = datasets_infer[key]
        elif key in spatial_keys:
            spatial_data[key] = datasets_infer[key]


    latitude = spatial_data['latitude']
    longitude = spatial_data['longitude']
    print(f'latitude shape {latitude.shape}')

    print('2. Split the data into x and y.')
    time_steps = datasets['AODANA'].shape[0]

    expanded_spatial_data = []
    for key in spatial_data.keys():
        expanded_data = np.repeat(spatial_data[key][np.newaxis, :, :], time_steps, axis=0)
        print(f'spatial key {key} has shape {expanded_data.shape}')
        masked_expanded_data = np.where(data_bm == 0, np.nan, expanded_data)
        expanded_spatial_data.append(masked_expanded_data)

    # Prepare feature data identified by feature_keys for stacking
    feature_keys = [key for key in datasets.keys() if key not in spatial_keys + ['MAIAC'] + ['time']]
    feature_data = [datasets[key] for key in feature_keys]
    
    # Stacking both feature data and expanded spatial data
    cv_data = np.stack(feature_data + expanded_spatial_data, axis=-1)
    y_data = datasets['MAIAC']  # Target variable
    
    print('3. Create the miss matrix')
    # Initialize the missing matrix with zeros
    miss_matrix = np.zeros_like(y_data, dtype=int)
    # Mark observed values as 1 (Observed AOD values are greater than 0 and not equal to -9999)
    observed_mask = (y_data != -1) & (y_data != -9999)
    miss_matrix[observed_mask] = 1
    
    print('4. Get the time dimension - get the season and month data')
    time_data = datasets_infer['time']
    date_arr = np.array([datetime.strptime(date,'%Y%m%d') for date in time_data])  
    # Extract month and season from the date array
    month_arr = np.array([date.month for date in date_arr])
    seasons = np.array([get_season(month) for month in month_arr])
    # For seasons - Map each season to an integer
    season_mapping = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
    season_arr = np.array([season_mapping[season] for season in seasons])

    print('6. Concatenate extra covariates to the X-variable')
    x_data = cv_data
    num_samples, height, width, channels = x_data.shape
    
    print('7. Perform scaling of x - train test and validation')
    x_data_scaled = np.empty_like(x_data, dtype=np.float32)

    for i in range(x_data_scaled.shape[-1]):
        channel_data = x_data[:, :, :, i]
        # Mask with NaN where binary mask is 0
        channel_data_masked = np.where(channel_data == -9999, np.nan, channel_data)
        # Flatten channel data, exclude NaN values for scaling
        valid_indices = ~np.isnan(channel_data_masked)
        valid_data = channel_data_masked[valid_indices].reshape(-1, 1)
        # Transform valid data using the loaded scaler
        scaled_valid_data = scalers[i].transform(valid_data)
        # Prepare an array filled with NaNs for the scaled channel
        scaled_channel = np.full(channel_data.shape, np.nan)
        # Place scaled data back, using valid indices
        scaled_channel[valid_indices] = scaled_valid_data.flatten()
        x_data_scaled[:, :, :, i] = scaled_channel

    y_data_model = np.expand_dims(y_data, axis=1)
    miss_matrix_model = np.expand_dims(miss_matrix, axis=1)
    
    # Transpose X data to match the model's expected input shape if necessary
    x_data_model = x_data_scaled.transpose(0, 3, 1, 2)  
    
    process_end_time = time.time()
    process_time_minutes = (process_end_time - process_start_time) / 60
    print(f'Execution time for pre-processing: {process_time_minutes:.2f} minutes')
    
    # Return the processed data ready for model inference
    return y_data_model, miss_matrix_model, x_data_model, date_arr, month_arr, season_arr, data_bm, latitude, longitude

y_train_model, y_test_model, y_val_model, miss_y_train_model, miss_y_test_model, miss_y_val_model, x_train_model, x_test_model, x_val_model, d_train, d_test, d_val, month_train, month_test, month_val, season_train, season_test, season_val, data_bm, scalers, latitude, longitude = get_data(hdf5_2021, hdf5_2022, koi, is_subset, aod_fv, fill_value)

# Saving X and Y datasets
with open(f'{base_dir}/Pickle/{region_abbr}_X_Train.pkl', 'wb') as f:
    pickle.dump([x_train_model], f)
 
with open(f'{base_dir}/Pickle/{region_abbr}_X_Test.pkl', 'wb') as f:
    pickle.dump([x_test_model], f)
 
with open(f'{base_dir}/Pickle/{region_abbr}_X_Validation.pkl', 'wb') as f:
    pickle.dump([x_val_model], f)

with open(f'{base_dir}/Pickle/{region_abbr}_Y_Train.pkl', 'wb') as f:
    pickle.dump([y_train_model, miss_y_train_model], f)
 
with open(f'{base_dir}/Pickle/{region_abbr}_Y_Test.pkl', 'wb') as f:
    pickle.dump([y_test_model, miss_y_test_model], f)
 
with open(f'{base_dir}/Pickle/{region_abbr}_Y_Validation.pkl', 'wb') as f:
    pickle.dump([y_val_model, miss_y_val_model], f)

# Saving date arrays
with open(f'{base_dir}/Pickle/{region_abbr}_DATE_TTV.pkl', 'wb') as f:
    pickle.dump([d_train, d_test, d_val], f)

with open(f'{base_dir}/Pickle/{region_abbr}_Season_TTV.pkl', 'wb') as f:
    pickle.dump([season_train, season_test, season_val], f)

with open(f'{base_dir}/Pickle/{region_abbr}_Month_TTV.pkl', 'wb') as f:
    pickle.dump([month_train, month_test, month_val], f)

# Saving binary mask
with open(f'{base_dir}/Pickle/{region_abbr}_BM.pkl', 'wb') as f:
    pickle.dump(data_bm, f)
     
with open(f'{base_dir}/Pickle/{region_abbr}_Grid.pkl', 'wb') as f:
    pickle.dump([latitude, longitude], f)
    
# After fitting the scalers to the training data
for i, scaler in enumerate(scalers):
    with open(f'{base_dir}/Pickle/X_scaler_{i}.pkl', 'wb') as f:
        pickle.dump(scaler, f)

infer_year = 2023
y_data_model, miss_matrix_model, x_data_model, date_arr, month_arr, season_arr, data_bm, latitude, longitude = get_data_inference(hdf5_2023, koi, is_subset, aod_fv, fill_value)