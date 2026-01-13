import numpy as np
import torch
from matplotlib import pyplot as plt

def histgram_of_single_day(mean_linear, week, ax, color):
    mean_linear_np = mean_linear.detach().numpy()

    # Normalize manually to (-1, 1)
    min_value = np.min(mean_linear_np)
    max_value = np.max(mean_linear_np)
    mean_linear_normalized = 2 * (mean_linear_np - min_value) / (max_value - min_value) - 1

    # Calculate histogram bins and counts
    n, bins, patches = ax.hist(mean_linear_normalized, bins=11, color=color)

    # Set edge colors
    for patch in patches:
        patch.set_edgecolor('black')  # Set bar edge color to black

    ax.set_title(f'Week {week}')
    ax.set_xlabel('Synaptic weights')
    ax.set_ylabel('Frequency')

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 800000)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([0, 400000, 800000])

# Process ANN models
def process_ann(phase1_list, week_dict,color):
    fig, axes = plt.subplots(nrows=1, ncols=len(phase1_list), figsize=(15, 5), sharey=True)
    for i, day in enumerate(phase1_list):
        model_path = f"./data/Figure-3-data/ann/s1_model_state_testday_{day}_turn_0_epoch_149.pth"
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            print(f"Successfully loaded ANN model parameters for day {day}!")
        except Exception as e:
            print(f"Failed to load ANN model parameters: {e}")
            exit()
        linear_weight = state_dict['encoder.fc_layers.1.weight']
        flatten_linear = linear_weight.flatten()
        histgram_of_single_day(flatten_linear, week_dict[day], axes[i], color)
    plt.tight_layout()
    plt.show()
    plt.close()

# Process SNN models
def process_snn(phase1_list, week_dict,color):
    fig, axes = plt.subplots(nrows=1, ncols=len(phase1_list), figsize=(15, 5), sharey=True)
    for i, day in enumerate(phase1_list):
        model_path = f"./data/Figure-3-data/snn/s1_model_state_testday_{day}_turn_0_epoch_149.pth"
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            print(f"Successfully loaded SNN model parameters for day {day}!")
        except Exception as e:
            print(f"Failed to load SNN model parameters: {e}")
            exit()
        linear_weight = state_dict['encoder.fc_layers.0.weight']
        flatten_linear = linear_weight.flatten()
        histgram_of_single_day(flatten_linear, week_dict[day], axes[i], color)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":

    phase1_list = [12, 15, 16, 19, 22]
    week_dict = {
        12: '1-4',
        15: '5',
        16: '6',
        19: '7',
        22: '8',
    }
    # Figure-3-i
    process_ann(phase1_list, week_dict,color='#DA357E')
    # Figure-3-n
    process_snn(phase1_list, week_dict,color='#31ACA6')

    