import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calc_metrics(dates, M_mb, Imputed_Y_mb, Y_mb):
    # Dictionary to store evaluation results for each date
    metrics_by_date = {}
    # Loop over each date to calculate the metrics per date
    for i, date in enumerate(dates):
        non_missing_mask  = M_mb[i]
        imputed_values = Imputed_Y_mb[i][non_missing_mask.bool()]
        observed_values = Y_mb[i][non_missing_mask.bool()]

        # Remove NaNs for correlation calculation
        valid_indices = ~np.isnan(observed_values.cpu().numpy()) & ~np.isnan(imputed_values.cpu().numpy())
        observed_values = observed_values.cpu().numpy()[valid_indices]
        imputed_values = imputed_values.cpu().numpy()[valid_indices]

        if len(observed_values) > 0 and len(imputed_values) > 0:
            # Evaluate the model's performance on the imputed values
            mae = mean_absolute_error(observed_values, imputed_values)
            mse = mean_squared_error(observed_values, imputed_values)
            rmse = np.sqrt(mse)
            
            # Check for zero variance
            if np.std(observed_values) != 0 and np.std(imputed_values) != 0:
                correlation = np.corrcoef(observed_values.flatten(), imputed_values.flatten())[0, 1]
            else:
                correlation = np.nan  # Handle the case where standard deviation is zero
        else:
            mae, mse, rmse, correlation = np.nan, np.nan, np.nan, np.nan

        # Save metrics for the current date
        if date not in metrics_by_date:
            metrics_by_date[date] = {'MAE': [], 'MSE': [], 'RMSE': [], 'Correlation': []}

        metrics_by_date[date]['MAE'].append(mae)
        metrics_by_date[date]['MSE'].append(mse)
        metrics_by_date[date]['RMSE'].append(rmse)
        metrics_by_date[date]['Correlation'].append(correlation)

    # Convert the metrics dictionary into a DataFrame
    df_metrics = {
        'Date': [],
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'Correlation': []
    }
    
    for date, metrics in metrics_by_date.items():
        df_metrics['Date'].append(date)
        df_metrics['MAE'].append(np.nanmean(metrics['MAE']))
        df_metrics['MSE'].append(np.nanmean(metrics['MSE']))
        df_metrics['RMSE'].append(np.nanmean(metrics['RMSE']))
        df_metrics['Correlation'].append(np.nanmean(metrics['Correlation']))
    
    df = pd.DataFrame(df_metrics)
    return df
