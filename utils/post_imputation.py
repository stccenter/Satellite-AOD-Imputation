import numpy as np

def process_imputed_data(imputed_data_samples, gain_data_samples, miss_matrix_model, y_raw_data, all_dates, bm):
    infer_concat = np.concatenate(imputed_data_samples, axis=0)
    gain_concat = np.concatenate(gain_data_samples, axis=0)
    y_data = np.concatenate(y_raw_data, axis = 0)
    
    total_days, channels, height, width = infer_concat.shape

    combined = list(zip(all_dates, infer_concat, gain_concat, y_data))
    combined.sort(key=lambda x: x[0])
    sorted_dates, sorted_infer_concat, sorted_gain_concat, sorted_y_data = zip(*combined)

    sorted_infer_concat = np.array(sorted_infer_concat)
    sorted_gain_concat = np.array(sorted_gain_concat)
    sorted_y_data = np.array(sorted_y_data)

    expanded_bm = np.expand_dims(np.expand_dims(bm, axis=0), axis=0)
    expanded_bm = np.repeat(expanded_bm, total_days, axis=0)
    
    imputed_aod = np.where(expanded_bm == 0, np.nan, sorted_infer_concat)
    gain_imputed_aod = np.where(expanded_bm == 0, np.nan, sorted_gain_concat)
    y_data = np.where((expanded_bm == 0)|(miss_matrix_model==0), np.nan, sorted_y_data)
    
    return imputed_aod, gain_imputed_aod, y_data, sorted_dates

def process_imputed_data_uq(uq_data_samples, gain_data_samples, miss_matrix_model, y_raw_data, all_dates, bm):
    uq_concat = np.concatenate(uq_data_samples, axis=0)
    gain_concat = np.concatenate(gain_data_samples, axis=0)
    y_data = np.concatenate(y_raw_data, axis = 0)

    
    total_days, channels, height, width = uq_concat.shape

    combined = list(zip(all_dates, uq_concat, gain_concat, y_data))
    combined.sort(key=lambda x: x[0])
    sorted_dates, sorted_uq_concat, sorted_gain_concat, sorted_y_data = zip(*combined)

    sorted_uq_concat = np.array(sorted_uq_concat)
    sorted_gain_concat = np.array(sorted_gain_concat)
    sorted_y_data = np.array(sorted_y_data)

    expanded_bm = np.expand_dims(np.expand_dims(bm, axis=0), axis=0)
    expanded_bm = np.repeat(expanded_bm, total_days, axis=0)
    
    imputed_aod = np.where(expanded_bm == 0, np.nan, sorted_uq_concat)
    gain_imputed_aod = np.where(expanded_bm == 0, np.nan, sorted_gain_concat)
    y_data = np.where((expanded_bm == 0)|(miss_matrix_model==0), np.nan, sorted_y_data)
    
    return imputed_aod, gain_imputed_aod, y_data, sorted_dates
