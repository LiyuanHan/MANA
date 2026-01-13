import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py

# Function to bin spike trains
def bin_spike(selected_trials, bin_size):
    num_bins = selected_trials.shape[1] // bin_size  # Calculate the number of complete bins
    remainder = selected_trials.shape[1] % bin_size  # Calculate the remaining columns

    # Initialize the binned matrix
    binned_matrix = torch.zeros((selected_trials.shape[0], num_bins + (1 if remainder > 0 else 0)))
    # Sum every bin_size columns
    for i in range(num_bins):
        start_col = i * bin_size
        end_col = start_col + bin_size
        binned_matrix[:, i] = torch.sum(selected_trials[:, start_col:end_col], dim=1)

        print(f"Original matrix shape: {selected_trials.shape}")
        print(f"Binned matrix shape: {binned_matrix.shape}")
    return binned_matrix

# Function to plot spike matrix
def plot_spike_matrix(spike_matrix, week_idx, classes):
    # Get the coordinates of non-zero points
    y, x = np.where(spike_matrix > 0)
    plt.scatter(x, y, c='black', s=1)
    plt.xlim(0, 27000)
    plt.xticks([0, 27000], ['0.6', '1.5'])
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    plt.title(f'Spike matrix of week {week_idx} target {classes}')
    filename = f'./figs/Fig2a-1.pdf'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# Function to plot mean firing rate heatmap
def mean_firerate_heat_map(mean_firing, week_idx):
    plt.figure(figsize=(12, 6))
    img = plt.imshow(mean_firing,
                     aspect='auto',
                     cmap="jet",  # Jet colormap
                     vmin=0, vmax=1, origin='lower')  # Fixed color range
    plt.xlim(0, 180)
    plt.xticks([0,180],['0.6', '1.5'])
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    plt.colorbar(img, label="Firing rate frequency")
    plt.title(f"Average firing rate frequency across trials week {week_idx}")
    plt.savefig(f"./figs/Fig2a-2.pdf")
    plt.show()


if __name__ == "__main__":
    '''
    We present the single-pulse-emission diagram for week 2 from Figure 2a, alongside the mean heatmap derived from 50 random trials.
    '''
    # Load data for week 2 (20221108 target 2 trial 121) and plot spike matrix
    data_trial = h5py.File('./data/Figure-2-data/Trial121.mat')
    spike_raw = torch.from_numpy(np.array(data_trial['spike_train']).T)
    target = int(torch.from_numpy(np.array(data_trial["target"])))
    plot_spike_matrix(spike_raw[:, :27000], 2, 2)

    # Load mean firing rate data for week 2 (20221108 target 2) and plot heatmap
    normalized_mean_firing = torch.load(
        './data/Figure-2-data/20221108_target_2_random_50_trials_mean_bin_spike_trains_normal.pt')
    mean_firerate_heat_map(normalized_mean_firing, '2')
