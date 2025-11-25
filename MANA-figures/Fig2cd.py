import numpy as np
import os
import sys


import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams["font.size"] = 8




print(" ")




## Prepare reading



cross_week_num = 1

models_list = [
    "MLP", 
    "GRU", 
    "Transformer",
    "CCA",
    "stabilization",
    "CEBRA", 
    "CycleGAN",
    "NoMAD",
    "MANAANN",
    "MANASNN",
]
models_label_dict = {
    "MLP": "MLP",
    "GRU": "GRU",
    "Transformer": "Transformer",
    "CCA": "CCA",
    "stabilization": "Stabilization",
    "CEBRA": "CEBRA",
    "CycleGAN": "Cycle-GAN",
    "NoMAD": "NoMAD",
    "MANAANN": "MANA (ANN)",
    "MANASNN": "MANA (SNN)",
}
models_label_list = [models_label_dict[model] for model in models_list]
colors_dict = {
    "MLP": '#33a9ff33', 
    "GRU": '#6baed680', 
    "Transformer": '#4292c6CD',  
    "CCA": "#f5e5f8CD", 
    "stabilization": "#dbccf2", 
    "CEBRA": "#c7bff7", 
    "CycleGAN": '#b395fc', 
    "NoMAD": "#a178df", 
    "MANAANN": '#da357eb3', 
    "MANASNN": '#00a29aa6'
}
colors_list = [colors_dict[model] for model in models_list]



import pickle

for phase in ["1", "2"]:

    if phase == "1":
        data_name = "Fig2c"
    elif phase == "2":
        data_name = "Fig2d"

    with open('data/' + data_name + '.pkl', 'rb') as f:
        acc_data_dict, len_data_dict = pickle.load(f)

    model_shape = acc_data_dict[models_list[0]].shape
    

    # sns boxplot

    import pandas as pd
    import seaborn as sns

    sns_labels = []
    for model_idx, model in enumerate(models_list):
        sns_labels += [f'{models_label_list[model_idx]}'] * len_data_dict[model]
    sns_accs = []
    for model in models_list:
        sns_accs.append(acc_data_dict[model].mean(1))

    sns_data = pd.DataFrame({
        "Models": sns_labels,
        "Accuracy (%)": np.concatenate(sns_accs),
    })

    f, ax = plt.subplots(figsize=(9, 3.5))
    plt.gcf().subplots_adjust(bottom=0.16, top=0.9)

    ax = sns.boxplot(
        # x=sns_data["Models"], y=sns_data["Accuracy (%)"],
        x=sns_data["Models"], y=sns_data["Accuracy (%)"], palette=colors_list,
        width=.5,
        showcaps=True, linewidth=1.5,
        showfliers=False,
    )

    ax.set_facecolor('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    plt.ylim(-5, 105)
    plt.yticks([0, 20, 40, 60, 80, 100])

    title = f"benchmark models, finetuned, phase {phase}"
    plt.title(title)

    fig_base_path = f"figs/"
    os.makedirs(fig_base_path, exist_ok=True)
    fig_name = f"{data_name}.pdf"

    plt.savefig(fig_base_path + fig_name)


