import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from utils.model_utils import sample_M
from utils.data_utils import convert_tr_dt_to_dt
from utils.loss import discriminator_loss, generator_loss
from utils.metrics import calc_metrics

def eval(loader, generator, discriminator, p_hint, alpha, beta, device):
    generator.eval()  # Set the generator to evaluation mode
    discriminator.eval()  # Set the discriminator to evaluation mode

    total_g_test_loss = 0
    total_d_test_loss = 0
    total_mse_test_loss = 0
    total_test_samples = 0
    
    imputed_test_data_samples = []
    gain_test_data_samples = []
    y_actual_samples = []
    date_list = []
    y_mask_list = []

    # Initialize an empty DataFrame to store all metrics
    all_metrics_df = pd.DataFrame()

    with torch.no_grad():  # No gradients needed for evaluation
        for batch_data in loader:
            
            X_mb = batch_data['X'].to(device)
            M_mb = batch_data['M'].to(device)
            Y_mb = batch_data['Y'].to(device)
            binary_mask = batch_data['binary_mask']
            month = batch_data['month']
            season = batch_data['season']
            lulc = batch_data['lulc']
            date_tensor = batch_data['date']
            dates = convert_tr_dt_to_dt(date_tensor)

            num_x_segments, x_channels, height, width = X_mb.shape
            num_y_segments, in_channels, height, width = Y_mb.shape

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

            D_loss, G_sample = discriminator_loss(generator, discriminator, X_mb, M_mb, New_Y_mb, H_mb, binary_mask_y_expanded, binary_mask_x_expanded, month, season, lulc)
            G_loss, Adv_loss, MSE_test_loss, MAE_test_loss, G_sample  = generator_loss(generator, discriminator, X_mb, Y_mb, M_mb, New_Y_mb, H_mb, binary_mask_y_expanded, binary_mask_x_expanded, month, season, lulc, alpha, beta)

            total_g_test_loss += G_loss.item()
            total_d_test_loss += D_loss.item()
            total_mse_test_loss += MSE_test_loss.item() * Y_mb.size(0)  # Accumulate MSE with weighting by batch size
            total_test_samples += Y_mb.size(0)

            # New_Y * ROI_mask, M * ROI_mask, X * ROI_mask, month, season
            Imputed_Y_mb = generator(New_Y_mb * binary_mask_y_expanded, M_mb * binary_mask_y_expanded, X_mb * binary_mask_y_expanded, month, season, lulc, binary_mask_y_expanded).detach()
            
            # Integrate imputed and observed data
            Final_Imputed_Y_mb = M_mb * Y_mb + (1 - M_mb) * Imputed_Y_mb
            
            # Apply the binary mask again to the final imputed data to ensure consistency within the ROI
            Final_Imputed_Y_mb *= binary_mask_y_expanded
            Imputed_Y_mb *= binary_mask_y_expanded
            
            #calculate metrics
            batch_metrics_df = calc_metrics(dates, M_mb, Imputed_Y_mb, Y_mb)
            
            # Append the current batch's metrics to the full DataFrame
            all_metrics_df = pd.concat([all_metrics_df, batch_metrics_df], ignore_index=True)
             
            imputed_test_data_samples.append(Final_Imputed_Y_mb.cpu().numpy()) 
            gain_test_data_samples.append(Imputed_Y_mb.cpu().numpy())
            y_actual_samples.append(Y_mb.cpu().numpy())
            y_mask_list.append(M_mb)
            date_list.extend(dates)
            
        avg_g_test_loss = total_g_test_loss / total_test_samples
        avg_d_test_loss = total_d_test_loss / total_test_samples
        avg_test_mse = total_mse_test_loss / total_test_samples
        avg_test_rmse = np.sqrt(total_mse_test_loss / total_test_samples) 

        # Print out the final average losses and total RMSE for training
        print(f'Average test G_loss: {avg_g_test_loss:.4f}')
        print(f'Average test D_loss: {avg_d_test_loss:.4f}')
        print(f'Average test MSE: {avg_test_mse:.4f}')
        print(f'Average test RMSE: {avg_test_rmse:.4f}')
        
        return imputed_test_data_samples, gain_test_data_samples, y_actual_samples, date_list, y_mask_list, all_metrics_df


def enable_dropout(model):
    """Function to enable dropout layers during evaluation."""
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()  

def eval_with_mcd(loader, generator, discriminator, p_hint, alpha, beta, z_score, device, num_samples=10):
    generator.eval()  # Set the generator to evaluation mode
    discriminator.eval()  # Set the discriminator to evaluation mode
    
    # Enable dropout layers during evaluation
    enable_dropout(generator)
    
    # total_g_test_loss = 0
    # total_d_test_loss = 0
    # total_mse_test_loss = 0
    # total_mae_test_loss = 0
    # total_test_samples = 0
    
    # imputed_data_samples = []

    gain_data_samples = []
    variance_data_samples = []
    lower_bound_samples = []
    upper_bound_samples = []
    y_actual_samples = []
    date_list = []
    y_mask_list = []

    # all_metrics_df = pd.DataFrame()

    with torch.no_grad():  # No gradients needed for evaluation
        for batch_data in loader:
            
            X_mb = batch_data['X'].to(device)
            M_mb = batch_data['M'].to(device)
            Y_mb = batch_data['Y'].to(device)
            binary_mask = batch_data['binary_mask']
            month = batch_data['month']
            season = batch_data['season']
            lulc = batch_data['lulc']
            date_tensor = batch_data['date']

            dates = convert_tr_dt_to_dt(date_tensor)

            num_x_segments, x_channels, height, width = X_mb.shape
            num_y_segments, in_channels, height, width = Y_mb.shape

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

            # Verify if dropout layers are in training mode
            for m in generator.modules():
                if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
                    print(f"Dropout layer {m} in training mode: {m.training}")

            # Perform multiple stochastic forward passes to capture uncertainty
            all_imputed_samples = []
            for i in range(num_samples):
                print(f'running the sample {i}')
                Imputed_Y_mb = generator(New_Y_mb * binary_mask_y_expanded, 
                                         M_mb * binary_mask_y_expanded, 
                                         X_mb * binary_mask_x_expanded, 
                                         month, season, lulc, binary_mask_y_expanded).detach()
                Imputed_Y_mb *= binary_mask_y_expanded  # Mask out outside ROI
                all_imputed_samples.append(Imputed_Y_mb)

            nan_mask = ~binary_mask_y_expanded.bool()  # Identify outside-ROI pixels

            # Set NaN before computing statistics
            all_imputed_samples[:, nan_mask] = float('nan')
            print(all_imputed_samples.shape)

            
            # Stack the predictions using torch.stack
            # all_imputed_samples = torch.stack(all_imputed_samples, dim=0)
            # # Calculate the mean and variance along the first dimension (stochastic passes)
            # mean_imputed_Y_mb = torch.mean(all_imputed_samples, dim=0)
            # variance_imputed_Y_mb = torch.var(all_imputed_samples, dim=0)
            # std_imputed_Y_mb = torch.sqrt(variance_imputed_Y_mb)

            # # Integrate imputed and observed data
            # # Final_Imputed_Y_mb = M_mb * Y_mb + (1 - M_mb) * mean_imputed_Y_mb
            # # Final_Imputed_Y_mb *= binary_mask_y_expanded  # Mask out pixels outside the ROI
            # # mean_imputed_Y_mb *= binary_mask_y_expanded 
            # # variance_imputed_Y_mb *= binary_mask_y_expanded
            # # std_imputed_Y_mb *= binary_mask_y_expanded
            
            # lower_bound = mean_imputed_Y_mb - z_score * std_imputed_Y_mb
            # upper_bound = mean_imputed_Y_mb + z_score * std_imputed_Y_mb
            
            # # Store the mean imputed values and uncertainty
            # # imputed_data_samples.append(Final_Imputed_Y_mb.cpu().numpy()) 
            # gain_data_samples.append(mean_imputed_Y_mb.cpu().numpy())
            # variance_data_samples.append(variance_imputed_Y_mb.cpu().numpy())
            # lower_bound_samples.append(lower_bound.cpu().numpy())
            # upper_bound_samples.append(upper_bound.cpu().numpy())
            # y_actual_samples.append(Y_mb.cpu().numpy())
            # y_mask_list.append(M_mb)
            # date_list.extend(dates)

            # Optionally, calculate any metrics (like RMSE) using the mean imputed values
            # batch_metrics_df = calc_metrics(dates, M_mb,mean_imputed_Y_mb, Y_mb.cpu())
            # all_metrics_df = pd.concat([all_metrics_df, batch_metrics_df], ignore_index=True)

            # total_test_samples += Y_mb.size(0)

    # After evaluation, compute overall losses (if required)
    # avg_g_test_loss = total_g_test_loss / total_test_samples
    # avg_d_test_loss = total_d_test_loss / total_test_samples
    # avg_test_mse = total_mse_test_loss / total_test_samples
    # avg_test_mae = total_mae_test_loss / total_test_samples
    # avg_test_rmse = np.sqrt(total_mse_test_loss / total_test_samples) 

    # print(f'Average test G_loss: {avg_g_test_loss:.4f}')
    # print(f'Average test D_loss: {avg_d_test_loss:.4f}')
    # print(f'Average test MSE: {avg_test_mse:.4f}')
    # print(f'Average test MAE: {avg_test_mae:.4f}')
    # print(f'Average test RMSE: {avg_test_rmse:.4f}')
    
    return gain_data_samples, variance_data_samples, lower_bound_samples, upper_bound_samples, y_actual_samples, y_mask_list, date_list 
