import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  

def create_xr(imputed_aod, date_arr, longitude, latitude, proj_string):
    x_prj = longitude[0,:]
    y_prj = latitude[:,0]
    date_xr = np.array(date_arr, dtype='datetime64[ns]')
    imputed_data = xr.Dataset(
        {
            "data": (["time", "latitude", "longitude"], imputed_aod[:,0,:])
        },
        coords={
            "time": ("time", date_xr),
            "longitude": ("longitude", x_prj),
            "latitude": ("latitude", y_prj),
        }
    )
    imputed_data.rio.write_crs(proj_string, inplace=True)
    return imputed_data
    
def save_output(output_path, imputed_data, save_mode = 1):
    print(type(imputed_data))
    if save_mode == 1:
        for time_idx in range(len(imputed_data.time)):
            # Select the data at this particular time
            single_time_slice = imputed_data.isel(time=time_idx)
            timestamp = pd.to_datetime(single_time_slice.time.values)
            year_doy = timestamp.strftime('%Y%j')  
            print(f'Saving {output_path}MAIAC_{year_doy}_GAIN.tif')
            tif_filename = f"{output_path}MAIAC_{year_doy}_GAIN.tif"
            single_time_slice.rio.to_raster(tif_filename)

    elif save_mode == 2:
        date_arr = imputed_data.time.values
        pd_date = pd.Timestamp(date_arr[0])
        year = pd_date.year
        nc4_filename = f"output_path{output_nc4}MAIAC_{year}_GAIN.nc"
        imputed_data.to_netcdf(path=output_filename)