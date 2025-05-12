import numpy as np
import matplotlib.pyplot as plt


def plot_subplot(fig, ax, data, title, cmap, norm=None):
    cax = ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title)
    fig.colorbar(cax, ax=ax, extend='both')

def create_plots(imputed_aod, gain_aod, y_data, miss_matrix_model, sorted_dates, date, cmap, norm = False):
    # Find the index for the given date
    idx = [i for i, dt in enumerate(sorted_dates) if dt == date]
    
    if len(idx) == 0:
        print(f"Date {date} not found in sorted_dates.")
        return

    idx = idx[0] 

    # Check if y_data is full NaN
    y_data_full_nan = np.isnan(y_data[idx, 0, :]).all()

    # Print the min and max values
    print("Merged imputed aod min:", np.nanmin(imputed_aod[idx, 0, :]), "max:", np.nanmax(imputed_aod[idx, 0, :]))
    print("GAIN imputed aod min:", np.nanmin(gain_aod[idx, 0, :]), "max:", np.nanmax(gain_aod[idx, 0, :]))
    print("Actual aod min:", np.nanmin(y_data[idx, 0, :]), "max:", np.nanmax(y_data[idx, 0, :]))

    max_val = np.nanmax(y_data[idx, 0, :])
    min_val = np.nanmin(y_data[idx, 0, :])

    # Create subplots with a single row and four columns
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    norm = plt.Normalize(min_val, max_val)

    # Plot imputed_aod without normalization if y_data is full NaN
    if y_data_full_nan:
        plot_subplot(fig, axs[0], imputed_aod[idx, 0, :], f'Merged Imputed AOD on {date}', cmap)
    else:
        if norm:
            plot_subplot(fig, axs[0], imputed_aod[idx, 0, :], f'Merged Imputed AOD on {date}', cmap, norm)
        else:
            plot_subplot(fig, axs[0], imputed_aod[idx, 0, :], f'Merged Imputed AOD on {date}', cmap)
    
    
    # Plot gain imputed_aod without normalization if y_data is full NaN
    if y_data_full_nan:
        plot_subplot(fig, axs[1], gain_aod[idx, 0, :], f'GAIN Imputed AOD on {date}', cmap)
    else:
        if norm:
            plot_subplot(fig, axs[1], gain_aod[idx, 0, :], f'GAIN Imputed AOD on {date}', cmap, norm)
        else:
            plot_subplot(fig, axs[0], gain_aod[idx, 0, :], f'GAIN Imputed AOD on {date}', cmap)

    # Plot y_data
    plot_subplot(fig, axs[2], y_data[idx, 0, :], f'Actual AOD with gaps on {date}', cmap)

    # Plot miss_matrix_model
    plot_subplot(fig, axs[3], miss_matrix_model[idx, 0, :], 'Miss Matrix Model', cmap)

    plt.tight_layout()
    plt.show()