import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from utils.model_utils import sample_M
from utils.data_utils import convert_tr_dt_to_dt
from utils.loss import discriminator_loss, generator_loss
from utils.metrics import calc_metrics


def run_inference(infer_loader, generator, discriminator, p_hint, alpha, beta, device):
    # Set the model to evaluation mode before inference
    generator.eval()
    discriminator.eval()
    
    total_g_test_loss = 0
    total_d_test_loss = 0
    total_mse_test_loss = 0
    total_mae_test_loss = 0
    total_test_samples = 0

    imputed_data_samples = []
    gain_data_samples = []
    y_raw_data = []
    y_mask_list = []
    all_dates = []

    # Initialize an empty DataFrame to store all metrics
    all_metrics_df = pd.DataFrame()
    
    with torch.no_grad():  # No gradients needed for inference
        for batch_data in infer_loader:
            Y_mb = batch_data['Y']
            X_mb = batch_data['X']
            M_mb = batch_data['M']
            binary_mask = batch_data['binary_mask']
            month = batch_data['month']
            season = batch_data['season']
            lulc = batch_data['lulc']
            date_tensor = batch_data['date']
    
            # Convert date tensor to numpy datetime
            dates = convert_tr_dt_to_dt(date_tensor)

            num_x_segments, x_channels, height, width = X_mb.shape
            num_y_segments, in_channels, height, width = Y_mb.shape
    
            # Add a channel dimension to the mask, making it [batch_size, 1, height, width]
            binary_mask_expanded = binary_mask.unsqueeze(1)
            binary_mask_x_expanded = binary_mask_expanded.expand(-1, x_channels, -1, -1)
            binary_mask_y_expanded = binary_mask.unsqueeze(1)
    
            # Generate random noise Z_mb with the same shape as your data
            Z_mb = torch.rand_like(Y_mb)
    
            # Generate hint matrix H_mb with a fraction of the observed data points hidden
            H_mb1 = sample_M(Y_mb.shape[0], Y_mb.shape[1:], 1 - p_hint)
            H_mb = M_mb * H_mb1
    
            # Combine the mask and the random noise to create New_X_mb
            New_Y_mb = M_mb * Y_mb + (1 - M_mb) * Z_mb  # Missing Data Introduction
    
            # Apply the ROI mask to ensure operations are only applied within the ROI
            New_Y_mb = New_Y_mb * binary_mask_y_expanded
            X_mb = X_mb * binary_mask_x_expanded
            M_mb = M_mb * binary_mask_y_expanded
            H_mb = H_mb * binary_mask_y_expanded

            D_loss, G_sample = discriminator_loss(generator, discriminator, X_mb, M_mb, New_Y_mb, H_mb, binary_mask_y_expanded, binary_mask_x_expanded, month, season, lulc)
            G_loss, Adv_loss, MSE_test_loss, MAE_test_loss, G_sample  = generator_loss(generator, discriminator, X_mb, Y_mb, M_mb, New_Y_mb, H_mb, binary_mask_y_expanded, binary_mask_x_expanded, month, season, lulc, alpha, beta)
    
            total_g_test_loss += G_loss.item()
            total_d_test_loss += D_loss.item()
            total_mse_test_loss += MSE_test_loss.item() * Y_mb.size(0)  # Accumulate MSE with weighting by batch size
            total_mae_test_loss += MAE_test_loss.item() * Y_mb.size(0)  
            total_test_samples += Y_mb.size(0)
    
            # Generate imputed data using the generator
            Imputed_Y_mb = generator(New_Y_mb, M_mb, X_mb, month, season, lulc, binary_mask_y_expanded).detach()
            
            # Integrate imputed and observed data
            Final_Imputed_Y_mb = M_mb * Y_mb + (1 - M_mb) * Imputed_Y_mb
            Final_Imputed_Y_mb *= binary_mask_y_expanded
            Imputed_Y_mb *= binary_mask_y_expanded

            batch_metrics_df = calc_metrics(dates, M_mb, Imputed_Y_mb, Y_mb)
            # Append the current batch's metrics to the full DataFrame
            all_metrics_df = pd.concat([all_metrics_df, batch_metrics_df], ignore_index=True)
            
            imputed_data_samples.append(Final_Imputed_Y_mb.cpu().numpy())
            gain_data_samples.append(Imputed_Y_mb.cpu().numpy())
            y_raw_data.append(Y_mb.cpu().numpy())
            y_mask_list.append(M_mb.cpu().numpy())
            all_dates.extend(dates)
            
        avg_g_test_loss = total_g_test_loss / total_test_samples
        avg_d_test_loss = total_d_test_loss / total_test_samples
        avg_test_mse = total_mse_test_loss / total_test_samples
        avg_test_mae = total_mae_test_loss / total_test_samples
        avg_test_rmse = np.sqrt(total_mse_test_loss / total_test_samples) 

        # Print out the final average losses and total RMSE for training
        print(f'Average test G_loss: {avg_g_test_loss:.4f}')
        print(f'Average test D_loss: {avg_d_test_loss:.4f}')
        print(f'Average test MSE: {avg_test_mse:.4f}')
        print(f'Average test MAE: {avg_test_mae:.4f}')
        print(f'Average test RMSE: {avg_test_rmse:.4f}')
        
        return imputed_data_samples, gain_data_samples, y_raw_data, all_dates, y_mask_list, all_metrics_df