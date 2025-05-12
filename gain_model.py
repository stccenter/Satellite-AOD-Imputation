import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
import torch.nn.init as init
import torch.optim as optim

from pyproj import CRS, Transformer

from datetime import datetime, timedelta
import time
from tqdm import tqdm

import os
import math
import numpy as np
import numpy.ma as ma
import pandas as pd

import xarray as xr
import rioxarray as rio
import pickle
import rasterio

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.data_utils import load_data, convert_tr_dt_to_dt, \
        process_lulc_data, remap_lulc_data, get_projection_string
from utils.model_utils import get_activation_function, get_optimizer
from utils.model_utils import sample_M
from utils.loss import generator_loss, discriminator_loss

# Test data
from utils.test_uq import eval, eval_with_mcd
# Infer data
from utils.inference import run_inference
from utils.post_imputation import process_imputed_data, process_imputed_data_uq
from utils.visuals import create_plots
from utils.create_raster import create_xr, save_output

import matplotlib.pyplot as plt
import tempfile

def training_loop():
    iteration = 0

    # Lists to store metrics for each mini-batch across all epochs
    g_train_loss_history = []
    d_train_loss_history = []
    mse_train_loss_history = []
    mae_train_loss_history = []

    # Train - Lists to store average metrics per epoch
    avg_g_train_loss_history = []
    avg_d_train_loss_history = []
    avg_mse_train_loss_history = []
    avg_mae_train_loss_history = []
    avg_rmse_train_loss_history = []

    #  Validation - Lists to store average metrics per epoch
    avg_g_val_loss_history = []
    avg_d_val_loss_history = []
    avg_mse_val_loss_history = []
    avg_mae_val_loss_history = []
    avg_rmse_val_loss_history = []

    for epoch in range(30):

        # Initialize variables to track progress
        total_g_train_loss = 0
        total_d_train_loss = 0
        total_mse_train_loss = 0
        total_mae_train_loss = 0
        total_train_samples = 0

        # train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i, batch_data in enumerate(train_loader):
            print('6. In model training phase')
            # Set models to training mode
            generator.train()
            discriminator.train()
    
            # Unpack the batch data
            Y_mb = batch_data['Y']  # observed data
            X_mb = batch_data['X'] #covariate
            M_mb = batch_data['M']  # The mask indicating observed data
            binary_mask = batch_data['binary_mask']
            month = batch_data['month']
            season = batch_data['season']
            lulc = batch_data['lulc']
            date_tensor = batch_data['date']
            
            dates = convert_tr_dt_to_dt(date_tensor)

            # First, add a channel dimension to the mask, making it [batch_size, 1, height, width]
            binary_mask_expanded = binary_mask.unsqueeze(1) 
            binary_mask_x_expanded = binary_mask_expanded.expand(-1, in_channels, -1, -1) 
            binary_mask_y_expanded = binary_mask.unsqueeze(1)
            
            # Generate random noise Z_mb with the same shape as your data
            Z_mb = torch.rand_like(Y_mb) 

            # Generate hint matrix H_mb1 with a fraction of the observed data points hidden
            H_mb1 = sample_M(Y_mb.shape[0], Y_mb.shape[1:], 1 - p_hint)
            H_mb = M_mb * H_mb1

            # Combine the mask and the random noise to create New_X_mb
            New_Y_mb = M_mb * Y_mb + (1 - M_mb) * Z_mb  # Missing Data Introduction

            # Apply the ROI mask to ensure operations are only applied within the ROI
            New_Y_mb = New_Y_mb * binary_mask_y_expanded
            Y_mb = Y_mb * binary_mask_y_expanded
            X_mb = X_mb * binary_mask_x_expanded
            M_mb = M_mb * binary_mask_y_expanded
            H_mb = H_mb * binary_mask_y_expanded

            # Transfer to device (if using GPU)
            if use_gpu:
                Y_mb = Y_mb.to("cuda")
                X_mb = X_mb.to("cuda")
                M_mb = M_mb.to("cuda")
                H_mb = H_mb.to("cuda")
                New_Y_mb = New_Y_mb.to("cuda")

            # Discriminator - Reset gradients and calculate losses and perform backpropagation
            optimizer_D.zero_grad()
            D_loss, G_sample = discriminator_loss(generator, discriminator, X_mb, M_mb, New_Y_mb, H_mb, binary_mask_y_expanded, binary_mask_x_expanded, month, season, lulc)
            D_loss.backward()

            # Zero out gradients for LULC embedding indices 0 and 255
            if generator.lulc_embedding.weight.grad is not None:
                generator.lulc_embedding.weight.grad[0].fill_(0) 
                
            optimizer_D.step()
            
            # Generator - Reset gradients and calculate losses and perform backpropagation
            optimizer_G.zero_grad()
            G_loss, Adv_loss, MSE_train_loss, MAE_train_loss, G_sample = generator_loss(generator, discriminator, X_mb, Y_mb, M_mb, New_Y_mb, H_mb, binary_mask_y_expanded, binary_mask_x_expanded, month, season, lulc, alpha, beta)       
            G_loss.backward()

            # Zero out gradients for LULC embedding indices 0 and 255
            if generator.lulc_embedding.weight.grad is not None:
                generator.lulc_embedding.weight.grad[0].fill_(0) 
            
            optimizer_G.step()

            # Training - Update running totals
            total_g_train_loss += G_loss.item()
            total_d_train_loss += D_loss.item()
            
            total_mse_train_loss += MSE_train_loss.item() * Y_mb.size(0) 
            total_mae_train_loss += MAE_train_loss.item() * Y_mb.size(0)
            total_train_samples += Y_mb.size(0)
            
            # Append the generator and discriminator losses for visualization
            g_train_loss_history.append(G_loss.item())
            d_train_loss_history.append(D_loss.item())
            mse_train_loss_history.append(MSE_train_loss.item())
            mae_train_loss_history.append(MAE_train_loss.item())
            
            iteration += 1

            print(f'Epoch {epoch} - batch {i} Generator loss: {G_loss.item():.4f}, D_loss: {D_loss.item():.4f}, Adversarial_train_loss: {Adv_loss.item():.4f}, MSE_train_loss: {MSE_train_loss.item():.4f}, MAE_train_loss: {MAE_train_loss.item():.4f}')

        # Calculate average losses for the epoch
        avg_g_train_loss = total_g_train_loss / total_train_samples
        avg_d_train_loss = total_d_train_loss / total_train_samples
        avg_train_mse = total_mse_train_loss / total_train_samples
        avg_train_mae = total_mae_train_loss / total_train_samples
        avg_train_rmse = np.sqrt(total_mse_train_loss / total_train_samples)
        
        # At the end of the epoch, append the averages
        avg_g_train_loss_history.append(avg_g_train_loss)
        avg_d_train_loss_history.append(avg_d_train_loss)
        avg_mse_train_loss_history.append(avg_train_mse)
        avg_mae_train_loss_history.append(avg_train_mae)
        avg_rmse_train_loss_history.append(avg_train_rmse)

        print(f'Epoch {epoch} | Average train G_loss: {avg_g_train_loss:.4f}, D_loss: {avg_d_train_loss:.4f}, MSE_train: {avg_train_mse:.4f}, MAE_train: {avg_train_mae:.4f}, RMSE_train: {avg_train_rmse:.4f}')

        print('===========================================================')
        print('7. In model validation phase')
        # Switch to evaluation mode for validation
        generator.eval()
        discriminator.eval()
        
        total_g_val_loss = 0
        total_d_val_loss = 0
        total_mse_val_loss = 0
        total_mae_val_loss = 0
        total_val_samples = 0
        
        with torch.no_grad():  # No gradients needed for evaluation
            for i, batch_data in enumerate(val_loader):
                print(f'epoch {epoch} batch data {i}')
                X_mb = batch_data['X'].to(device)
                M_mb = batch_data['M'].to(device)
                Y_mb = batch_data['Y'].to(device)
                binary_mask = batch_data['binary_mask']
                month = batch_data['month']
                season = batch_data['season']
                lulc = batch_data['lulc']

                # First, add a channel dimension to the mask, making it [batch_size, 1, height, width]
                binary_mask_expanded = binary_mask.unsqueeze(1) 
                binary_mask_x_expanded = binary_mask_expanded.expand(-1, x_channels, -1, -1) 
                binary_mask_y_expanded = binary_mask.unsqueeze(1)

                # Generate random noise Z_mb with the same shape as your data
                Z_mb = torch.rand_like(Y_mb)

                # Generate hint matrix H_mb1 with a fraction of the observed data points hidden
                H_mb1 = sample_M(Y_mb.shape[0], Y_mb.shape[1:], 1 - p_hint)
                H_mb = M_mb * H_mb1

                # Combine the mask and the random noise to create New_X_mb
                New_Y_mb = M_mb * Y_mb + (1 - M_mb) * Z_mb  # Missing Data Introduction

                # Apply the ROI mask to ensure operations are only applied within the ROI
                New_Y_mb = New_Y_mb * binary_mask_y_expanded
                X_mb = X_mb * binary_mask_x_expanded
                M_mb = M_mb * binary_mask_y_expanded
                H_mb = H_mb * binary_mask_y_expanded

                # Discriminator loss
                D_loss, G_sample = discriminator_loss(generator, discriminator, X_mb, M_mb, New_Y_mb, H_mb, binary_mask_y_expanded, binary_mask_x_expanded, month, season, lulc)
                # Generator loss
                G_loss, Adv_loss, MSE_val_loss, MAE_val_loss, G_sample = generator_loss(generator, discriminator, X_mb, Y_mb, M_mb, New_Y_mb, H_mb, binary_mask_y_expanded, binary_mask_x_expanded, month, season, lulc, alpha, beta)

                # Validation - Update running totals
                total_g_val_loss += G_loss.item()
                total_d_val_loss += D_loss.item()
                total_mse_val_loss += MSE_val_loss.item() * Y_mb.size(0)  
                total_mae_val_loss += MAE_val_loss.item() * Y_mb.size(0)
                total_val_samples += Y_mb.size(0)
                
            avg_g_val_loss = total_g_val_loss / total_val_samples
            avg_d_val_loss = total_d_val_loss / total_val_samples
            avg_val_mse = total_mse_val_loss / total_val_samples
            avg_val_mae = total_mae_val_loss / total_val_samples
            avg_val_rmse = np.sqrt(total_mse_val_loss / total_val_samples) 
            avg_val_mae = total_mae_val_loss / total_val_samples
            
            # At the end of the epoch, append the averages
            avg_g_val_loss_history.append(avg_g_val_loss)
            avg_d_val_loss_history.append(avg_d_val_loss)
            avg_mse_val_loss_history.append(avg_val_mse)
            avg_mae_val_loss_history.append(avg_val_mae)
            avg_rmse_val_loss_history.append(avg_val_rmse)

            print(f'Epoch {epoch} | Average val G_loss: {avg_g_val_loss:.4f}, D_loss: {avg_d_val_loss:.4f},MSE_val: {avg_val_mse:.4f}, MAE_val: {avg_val_mae:.4f}, RMSE_val: {avg_val_rmse:.4f}')
            print('\n')

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, xdata, ydata, mask, binary_mask, month_data, season_data):
        self.xdata = torch.tensor(np.nan_to_num(xdata, nan=-1.0), dtype=torch.float32)
        self.ydata = torch.tensor(np.nan_to_num(ydata, nan=-1.0), dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.binary_mask = torch.tensor(binary_mask, dtype=torch.float32)
        self.month_data = torch.tensor(month_data, dtype=torch.long)  
        self.season_data = torch.tensor(season_data, dtype=torch.long)
        # self.lulc_data = torch.tensor(lulc_data, dtype=torch.long)
        
    def __len__(self):
        return len(self.ydata)

    def __getitem__(self, idx):
         return {
                    'X': self.xdata[idx],
                    'Y': self.ydata[idx],
                    'M': self.mask[idx],
                    'binary_mask': self.binary_mask,
                    'month': self.month_data[idx], 
                    'season': self.season_data[idx]
                    # 'lulc': self.lulc_data
                }
        
def stream_data_loader(dataset, batch_size=1, shuffle=True):
    worker_init_fn = lambda _: np.random.seed()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, worker_init_fn=worker_init_fn)
    while True:
        for data in dataloader:
            yield data


class CustomIterableDataset(IterableDataset):
    def __init__(self, xdata, ydata, mask, binary_mask, month_data, season_data, lulc_data, date_data):
        # lulc_data
        self.xdata = torch.tensor(np.nan_to_num(xdata, nan=0.0), dtype=torch.float32)
        self.ydata = torch.tensor(np.nan_to_num(ydata, nan=0.0), dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.binary_mask = torch.tensor(binary_mask, dtype=torch.float32)
        # self.binary_mask = torch.tensor(binary_mask, dtype=torch.bool)
        self.month_data = torch.tensor(month_data, dtype=torch.long)
        self.season_data = torch.tensor(season_data, dtype=torch.long)
        self.lulc_data = torch.tensor(lulc_data, dtype=torch.long)
        self.date_data = np.array([date.timestamp() for date in date_data])  
        
    def process_item(self, idx):
        return {
            'X': self.xdata[idx],
            'Y': self.ydata[idx],
            'M': self.mask[idx],
            'binary_mask': self.binary_mask,
            'month': self.month_data[idx], 
            'season': self.season_data[idx],
            'date': self.date_data[idx],
            'lulc': self.lulc_data
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  
            iter_start = 0
            iter_end = len(self.ydata)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((len(self.ydata) / float(worker_info.num_workers))))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.ydata))

        return (self.process_item(idx) for idx in range(iter_start, iter_end))
    

class CustomLULCEmbedding(nn.Module):
    def __init__(self, num_lulc_classes, lulc_embed_dim):
        super().__init__()
        self.lulc_embedding = nn.Embedding(num_lulc_classes, lulc_embed_dim)
        # Explicitly zero out the first embedding vector which is typically used for padding
        self.lulc_embedding.weight.data[0].fill_(0)

    def forward(self, x):
        return self.lulc_embedding(x)

    @property
    def weight(self):
        return self.lulc_embedding.weight

    
class Generator(nn.Module):
    def __init__(self, in_channels, covariate_channels, activation_fn, 
                 num_lulc_classes, lulc_embed_dim=256,
                 num_months=12, month_embed_dim=24, 
                 num_seasons=4, season_embed_dim=8, 
                 dropout_rate = 0.2):

        super(Generator, self).__init__()
        
        # Embedding for months
        self.month_embedding = nn.Embedding(num_months, month_embed_dim)
        # Embedding for seasons
        self.season_embedding = nn.Embedding(num_seasons, season_embed_dim)
        # Embedding for LULC       
        self.lulc_embedding = CustomLULCEmbedding(num_lulc_classes, lulc_embed_dim)
        print('in, covariate, month, season, lulc')
        print(in_channels, covariate_channels, month_embed_dim, season_embed_dim, lulc_embed_dim) 
        total_channels = in_channels + 1 + covariate_channels   + month_embed_dim + season_embed_dim + lulc_embed_dim
        
        # Define the generator architecture with convolutional, batch normalization, and ReLU layers in sequence
        self.gen = nn.Sequential(
            nn.Conv2d(in_channels=total_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.Dropout2d(dropout_rate), 

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.Dropout2d(dropout_rate), 

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.2, inplace=True),
            nn.Dropout2d(dropout_rate), 

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Softplus()
        )

    def forward(self, corrupted_data, mask, covariates,  months, seasons, lulc_data, y_mask):
        # Extract height and width from corrupted_data
        height = corrupted_data.size(2)
        width = corrupted_data.size(3)

        # Ensure y_mask is of type long for compatibility with embedding tensors
        y_mask = y_mask.to(torch.long)
         
        # Get month embeddings
        month_embeddings = self.month_embedding(months)  # Shape: [batch_size, month_embed_dim]
        month_embeddings = month_embeddings.view(-1, month_embeddings.shape[1], 1, 1).repeat(1, 1, height, width)

        # Get season embeddings
        season_embeddings = self.season_embedding(seasons)  # Shape: [batch_size, season_embed_dim]
        season_embeddings = season_embeddings.view(-1, season_embeddings.shape[1], 1, 1).repeat(1, 1, height, width)

        # Get LULC embeddings
        lulc_embeddings = self.lulc_embedding(lulc_data)
        lulc_embeddings = lulc_embeddings.permute(0, 3, 1, 2) 

        # Concatenate mask and covariates with corrupted data along the channel dimension
        x = torch.cat([corrupted_data, mask, covariates, month_embeddings, season_embeddings, lulc_embeddings], dim=1)
        # Pass the concatenated input through the generator network
        x = self.gen(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, activation_fn, dropout_rate = 0.1):
        super(Discriminator, self).__init__()

        # Define the discriminator architecture with convolutional, batch normalization, and ReLU layers in sequence
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride = 1, padding=1),
            activation_fn,
            nn.Dropout2d(dropout_rate), 

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride = 1, padding=1),
            activation_fn,
            nn.Dropout2d(dropout_rate), 

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride = 1, padding=1),
            activation_fn,
            nn.Dropout2d(dropout_rate), 
            
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride = 1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, combined_input, hint):
        # Concatenate the combined input and hint along the channel dimension
        x = torch.cat([combined_input, hint], dim=1)
        # Pass the concatenated input through the discriminator network
        x = self.disc(x)
        return x


abbr_dict_region = {'Northeast': 'NE',
             'Upper Midwest': 'UM',
             'Ohio Valley': 'OV',
             'Southeast': 'SE',
             'South': 'S',
             'Southwest':'SW',
             'Northern Rockies': 'NR',
             'West': 'W',
             'Northwest': 'NW'}


if __name__ == "__main__":
        region_name = 'Northeast'
        region_abbr = abbr_dict_region.get(region_name)

        # Read and process LULC data
        lulc_filepath = f'./data/LULC/{region_abbr}_LULC.tif'
        output_pkl =  f'./data/Pickle/'
        models_path = f'./data/model/'

        print("Loading datasets...")

        x_train_model = load_data(f'{output_pkl}/{region_abbr}_X_Train.pkl')
        x_train_model = x_train_model[0]
        y_train_model, miss_y_train_model = load_data(f'{output_pkl}/{region_abbr}_Y_Train.pkl')
        y_train_model = np.where((y_train_model == -9999) | (y_train_model == -1.0), np.nan, y_train_model)

        x_val_model = load_data(f'{output_pkl}/{region_abbr}_X_Validation.pkl')
        x_val_model = x_val_model[0]
        y_val_model, miss_y_val_model = load_data(f'{output_pkl}/{region_abbr}_Y_Validation.pkl')
        y_val_model = np.where((y_val_model == -9999) | (y_val_model == -1.0), np.nan, y_val_model)

        x_test_model = load_data(f'{output_pkl}/{region_abbr}_X_Test.pkl')
        x_test_model = x_test_model[0]
        y_test_model, miss_y_test_model = load_data(f'{output_pkl}/{region_abbr}_Y_Test.pkl')
        y_test_model = np.where((y_test_model == -9999) | (y_test_model == -1.0), np.nan, y_test_model)

        d_train, d_test, d_val = load_data(f'{output_pkl}/{region_abbr}_DATE_TTV.pkl')
        season_train, season_test, season_val = load_data(f'{output_pkl}/{region_abbr}_Season_TTV.pkl')
        month_train, month_test, month_val = load_data(f'{output_pkl}/{region_abbr}_Month_TTV.pkl')
        data_bm = load_data(f'{output_pkl}/{region_abbr}_BM.pkl')
        latitude, longitude = load_data(f'{output_pkl}/{region_abbr}_Grid.pkl')
        # y_scaler = load_data(f'{output_pkl}Y_scaler.pkl')
        print("Datasets loaded.")

        # Read LULC
        lulc_data, class_to_index = process_lulc_data(lulc_filepath)
        lulc_data_remapped = remap_lulc_data(lulc_data, class_to_index)

        lulc_remap_values = np.unique(lulc_data_remapped, return_counts = True)
        lulc_actual_values = np.unique(lulc_data, return_counts = True)

        num_lulc_classes = len(lulc_remap_values[0])
        lulc_data_remapped_int = lulc_data_remapped.astype(np.int32)

        # Adjust month data and create datasets
        adj_month_train = month_train - 1
        adj_month_val = month_val - 1
        adj_month_test = month_test - 1

        train_dataset = CustomIterableDataset(x_train_model, y_train_model, miss_y_train_model, data_bm[0], adj_month_train, season_train, lulc_data_remapped_int, d_train)
        val_dataset = CustomIterableDataset(x_val_model, y_val_model, miss_y_val_model, data_bm[0], adj_month_val, season_val, lulc_data_remapped_int, d_val)
        test_dataset = CustomIterableDataset(x_test_model, y_test_model, miss_y_test_model, data_bm[0], adj_month_test, season_test, lulc_data_remapped_int, d_test)
        print("Training and validation datasets prepared.")

        p_hint = 0.2
        batch_size = 32
        num_epochs = 50
        alpha = 0.05
        beta = 0.05
        activation_fn = get_activation_function('LeakyReLU')

        optimizer_name_G = "AdamW"
        optimizer_name_D = "SGD"
        learning_rate_G = 0.00001
        learning_rate_D = 0.00001
        g_weight_decay = 1e-5
        d_weight_decay = 1e-5
        betas = [0.5, 0.999]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,num_workers=2)

        generator_path = f'{models_path}/generator_model_v1.pth' 
        discriminator_path = f'{models_path}/discriminator_model_v1.pth' 

        print('1. Get the input shape dimensions')

        # Get the first batch from the DataLoader
        first_batch = next(iter(test_loader))
        x_first_batch, y_first_batch = first_batch['X'], first_batch['Y']

        # Extracting the shape of your input data
        num_y_segments, in_channels, height, width = y_first_batch.shape
        num_x_segments, x_channels, height, width = x_first_batch.shape

        print('2. Set the model')
        # Instantiate the Generator
        generator = Generator(in_channels, x_channels, activation_fn, num_lulc_classes)   

        # Instantiate the Discriminator
        discriminator = Discriminator(in_channels, activation_fn = activation_fn)

        print('3. Set the device')
        use_gpu = torch.cuda.is_available()  # Automatically check if GPU is available
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"


        torch.manual_seed(42)  # Set seed for CPU operations.
        generator.to(device)
        discriminator.to(device)

        print('4. Set optimizer')
        optimizer_G = get_optimizer(optimizer_name_G, generator.parameters(), learning_rate_G, g_weight_decay, betas, momentum=0.9)
        optimizer_D = get_optimizer(optimizer_name_D, discriminator.parameters(), learning_rate_D, d_weight_decay, betas, momentum=0.9)

        training_loop()