import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams["font.size"] = 8

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd



print(" ")




## Prepare reading





for model_category in ["benchmark", "manifold"]:

    if model_category == "benchmark":
        models_list = [
            "MLP", 
            "GRU", 
            "Transformer",
            "MANAANN",
            "MANASNN",
        ]
        ann_tukey_indices = [5, 0, 6]
        snn_tukey_indices = [7, 1, 8]
    elif model_category == "manifold":
        models_list = [
            "CCA",
            "stabilization",
            "CEBRA", 
            "CycleGAN",
            "NoMAD",
            "MANAANN",
            "MANASNN",
        ]
        ann_tukey_indices = [2, 17, 7, 11, 16]
        snn_tukey_indices = [3, 19, 8, 12, 18]
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

    if model_category == "benchmark":
        data_name = "Fig2g1"
    elif model_category == "manifold":
        data_name = "Fig2g2"

    with open('data/' + data_name + '.pkl', 'rb') as f:
        acc_data_dict, len_data_dict = pickle.load(f)

    model_shape = acc_data_dict[models_list[0]].shape
        

    # ANOVA

    from scipy.stats import f_oneway

    anova_p_val_list = []

    for day_idx in range(len(acc_data_dict[models_list[0]])):
        data_groups = [acc_data_dict[model][day_idx] for model in models_list]
        f_stat, p_val = f_oneway(*data_groups)
        print(f"ANOVA for day index {day_idx}: f-stat = {f_stat:.2f}, p-val = {p_val}")
        anova_p_val_list.append(p_val)
    print(" ")

    def convert_p_val_to_label(pvalue):
        p_val_thres_list = [0.0001, 0.001, 0.01, 0.05]
        label_list = ["****", "***", "**", "*"]
        for p_val_thres, label in zip(p_val_thres_list, label_list):
            if pvalue < p_val_thres:
                return label
        return "n.s."

    # barplot

    bar_width = 1.0 / (len(models_list) + 1.5)
    x_pos_correction = [i - (len(models_list) - 1) / 2 for i in range(len(models_list))]

    f, ax = plt.subplots(figsize=(12, 3.5))
    # plt.gcf().subplots_adjust(bottom=0.16, top=0.9)

    for model_idx, model in enumerate(models_list):
        x_pos = np.array(list(range(len_data_dict[model] + 2))) + x_pos_correction[model_idx] * bar_width
        ax.bar(
            x_pos, acc_data_dict[model].mean(1), yerr=acc_data_dict[model].std(1) / np.sqrt(model_shape[1]),
            width=bar_width, color=colors_list[model_idx], linewidth=.5, edgecolor='#ffffff', capsize=2,
            align='center', label=models_label_list[model_idx],
        )

    x_pos = np.array(list(range(len_data_dict[models_list[0]] + 2)))
    for day_idx, x in enumerate(x_pos):
        ax.text(x, 100, convert_p_val_to_label(anova_p_val_list[day_idx]), ha='center')

    x_labels = ['-2', '0', '1', '4', '6', '8', '11', '14', '18', '25', '26', '27', '31']
    plt.xlabel('Decoding day (Replacing phase)')
    plt.xticks(list(range(len_data_dict[models_list[0]] + 2)), x_labels)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    plt.yticks([0, 20, 40, 60, 80, 100])

    plt.legend(models_label_list, loc='lower right')

    title = f"benchmark models, finetuned, phase R"
    plt.title(title)

    fig_base_path = f"figs/"
    os.makedirs(fig_base_path, exist_ok=True)
    fig_name = f"{data_name}.pdf"

    plt.savefig(fig_base_path + fig_name)
    
