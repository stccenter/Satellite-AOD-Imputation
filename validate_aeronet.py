import glob, sys, os
from datetime import datetime
sys.path.append('/opt/AQ/UseCase/Thesis/Script/Utils/')
from GeoDataManager import GeoDataManager
import CollocateManager as utils
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS
import rioxarray as rio
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, linregress, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from CZ_Regions import CZ_Regions
from calendar import isleap


def show_scatterplot(accumulated_combined_df):

    accumulated_combined_df = accumulated_combined_df.dropna(subset=['AERONET_AOD', 'Averaged_Sat_AOD'])

    # Extract AERONET and MODIS AOD values
    x = accumulated_combined_df['AERONET_AOD'].values
    y = accumulated_combined_df['Averaged_Sat_AOD'].values


    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = z / z.max()

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    # Compute metrics
    slope = np.sum(x * y) / np.sum(x ** 2)  # Least squares with zero intercept
    r, _ = pearsonr(x, y)
    rmse = np.sqrt(mean_squared_error(y, x))
    mae = mean_absolute_error(y, x)
    rmb = np.mean(y) / np.mean(x)
    bias = np.mean(y - x)


    # Compute EE percentages
    count_within_ee = np.sum((x - (0.05 + 0.20 * x) <= y) & (y <= x + (0.05 + 0.20 * x)))
    ee_percentage = 100 * count_within_ee / len(x)
    above_ee = 100 - ee_percentage - np.sum(y < x - (0.05 + 0.20 * x)) / len(x) * 100
    below_ee = 100 - ee_percentage - above_ee
    print(rmse, mae, rmb, bias, ee_percentage)
        
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with density coloring
    scatter = ax.scatter(x, y, c=z, s=8, edgecolor=None, cmap='jet')

    # Add colorbar
#     cbar = plt.colorbar(scatter, ax=ax)
#     cbar.set_label('Density')

    # Perform linear regression
    line_x = np.linspace(min(x), max(x), 1000) 
    line_y = slope * line_x  # No intercept
    ax.plot(line_x, line_y, color='black')
    
    # Set equal range for both axes
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 3.5)

    # Extended Error (EE) lines
    x_range = np.linspace(0, max(x), 100)  # Ensure it starts from 0
    ax.plot(x_range, x_range * 1.2, linestyle='dashed', color='grey')  # Upper EE line
    ax.plot(x_range, x_range * 0.8, linestyle='dashed', color='grey') 

    # 1:1 line
    ax.plot(x_range, x_range, color='red')


    # Labels and title
    ax.set_xlabel('AERONET AOD L2 (550nm)', fontsize = 20)
    ax.set_ylabel('MODIS MAIAC (550nm)', fontsize = 20)
    ax.tick_params(axis='both', labelsize=15)

    
    # Add metrics to the top-left corner
    metrics_text = f'Y={slope:.2f}X\nN={len(x)}\nR={r:.2f}\n' \
        f'RMSE={rmse:.2f}\n'\
        f'MAE={mae:.2f}\n'\
        f'RMB={rmb:.2f}\n'\
        f'Bias={bias:.2f}' 


    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, fontsize=15,
        verticalalignment='top')
    
    metrics_text1 = f'{"Within EE":<10} {ee_percentage:.2f}%' 

    
    ax.text(0.28, 0.95, metrics_text1, transform=ax.transAxes, fontsize=15,
        verticalalignment='top')

    # Show plot
    plt.show()

accumulated_combined_df = pd.read_csv(f'./data/AERONET/NE_AERONET_MAIAC_2023.csv')
show_scatterplot(accumulated_combined_df)