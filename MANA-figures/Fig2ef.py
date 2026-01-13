import matplotlib.pyplot as plt
import torch

def load_data_from_pt(filename="./data/Figure-2-data/Figure-2-e-data.pt"):
    """Load data from a .pt file."""
    return torch.load(filename)

if __name__ == "__main__":
    '''
    Figure 2-e
    '''
    loaded_data = load_data_from_pt()

    # Set model names and colors
    models = list(loaded_data["AC"].keys())
    colors = ['#0094FF', '#FF9200', '#008D00', '#9F96A3', '#DA357E', '#31ACA6']

    # 创建横向排列子图
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 调整为 1 行 3 列

    # Plot each subplot
    for i, (operation, values) in enumerate(loaded_data.items()):
        # Prepare data
        model_names = list(values.keys())
        values_list = list(values.values())

        # Create bar chart
        axs[i].bar(model_names, values_list, color=colors)

        # Set title and labels
        axs[i].set_title(operation)
        axs[i].set_ylabel('Computational cost (uJ)')
        axs[i].set_xlabel('Decoding algorithms')

        # 调整x轴刻度字体大小
        axs[i].tick_params(axis='x', labelsize=8)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()
    plt.close()


    '''
    Figure 2-f
    '''
    # Load data from .pt file
    loaded_data = load_data_from_pt(filename="./data/Figure-2-data/Figure-2-f-data.pt")

    # Configuration
    plt.style.use('seaborn-v0_8-poster')
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    operations = loaded_data["operations"]
    colors = ['#0094FF', '#008D00', '#DA357E', '#31ACA6']

    # Plot for each operation
    for i, operation in enumerate(operations):
        ax = axs[i]
        width = 0.15  # Reduced width for more spacing

        # Plot Dimensionality Reduction (same for ANN and SNN)
        dim_red = ax.bar(-0.3, loaded_data["Dimensionality Reduction"][i], width,
                         label='Dimensionality reduction', color=colors[0])

        # Plot Alignment (same for ANN and SNN)
        align = ax.bar(-0.1, loaded_data["Alignment"][i], width,
                       label='Manifold alignment', color=colors[1])

        # Plot ANN Adaptation
        ann_adapt = ax.bar(0.1, loaded_data["ANN_Adaptation"][i], width,
                           label='Network adaptation (ANN)', color=colors[2])

        # Plot SNN Adaptation
        snn_adapt = ax.bar(0.3, loaded_data["SNN_Adaptation"][i], width,
                           label='Network adaptation (SNN)', color=colors[3])

        # Configure axes
        ax.set_title(f'({operation})')
        ax.set_ylabel('Computational cost (uJ)', fontsize=10)
        ax.set_xlabel('Module in MANA (ANN/SNN)', fontsize=10)
        ax.set_xticks([-0.3, -0.1, 0.1, 0.3])  # Set four tick positions
        ax.set_xticklabels([])  # Remove tick labels
        ax.legend(loc='lower left', fontsize=8)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    plt.close()

    