import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams["font.size"] = 8

import pandas as pd
import scipy
import seaborn as sns
sns.set_theme(style="white")

import pickle


print(" ")




jitter = 0

labels_dict = {
    'MANAANN': '4 models',
    'MANAANN-UOM': '1 model',
    'MANASNN': '4 models',
    'MANASNN-UOM': '1 model',
}
colors_dict = {
    "MANAANN": '#da357eb3', 
    'MANAANN-UOM': "#d85791b3",
    "MANASNN": '#00a29aa6',
    "MANASNN-UOM": "#48cac3a6",
}
model_nums_dict = {
    "MANAANN": 4, 
    'MANAANN-UOM': 1,
    "MANASNN": 4,
    "MANASNN-UOM": 1,
}

for phase in ["1", "2"]:

    if phase == "1":
        abnormal_threshold = 80
        data_name_ = "FigEx3a"
    elif phase == "2":
        abnormal_threshold = 40
        data_name_ = "FigEx3b"

    with open('data/' + data_name_ + '.pkl', 'rb') as f:
        all_acc = pickle.load(f)


    for models_list in [['MANAANN', 'MANAANN-UOM'], ['MANASNN', 'MANASNN-UOM']]:

        if 'ANN' in models_list[0]:
            data_name = data_name_ + '1'
        elif 'SNN' in models_list[0]:
            data_name = data_name_ + '2'

        acc_pair = [0, 0]

        p_val_data_dict = dict()
        p_val_data_list = [None] * len(models_list)
        box_plot_data_dict = dict()
        box_plot_data_list = [None] * len(models_list)

        for model_idx, model in enumerate(models_list):

            model_acc = all_acc[model]
            daily_acc = model_acc.mean(axis=1)
            if model_idx == 0:
                day_indices = list(idx for idx in range(len(daily_acc)) if daily_acc[idx] <= abnormal_threshold)
            else:
                pass

            daily_acc = np.delete(daily_acc, day_indices)
            phase_median = np.median(daily_acc)
            acc_pair[model_idx] = phase_median
            
            p_val_data_dict[model] = pd.DataFrame({
                'Number of models': np.repeat(model, len(daily_acc)),
                'Accuracy (%)': daily_acc,
            })
            p_val_data_list[model_idx] = daily_acc.copy()
            box_plot_data_dict[model] = pd.DataFrame({
                'Number of models': np.repeat(model_nums_dict[model], model_acc.shape[0]),
                'Accuracy (%)': model_acc.mean(1),
            })
            box_plot_data_list[model_idx] = model_acc.mean(1).copy()
            
            daily_sem = np.std(model_acc, axis=1) / np.sqrt(model_acc.shape[1])

            
        t_stat_r, p_val_r = scipy.stats.ttest_rel(p_val_data_dict[models_list[0]]['Accuracy (%)'], p_val_data_dict[models_list[1]]['Accuracy (%)'],)
        p_val_r = (p_val_r / 2) if t_stat_r > 0 else (1 - p_val_r / 2)


        def convert_p_val_to_label(pvalue):
            p_val_thres_list = [0.001, 0.01, 0.05]
            for i, p_val_thres in enumerate(p_val_thres_list):
                if pvalue < p_val_thres:
                    return f"p<{p_val_thres}"
            return "n.s."
        

        f, ax = plt.subplots()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        p_val_data = pd.concat([p_val_data_dict[model] for model in reversed(models_list)])
        box_plot_data = pd.concat([box_plot_data_dict[model] for model in reversed(models_list)])

        data = box_plot_data
        data_list = np.array(list(reversed(box_plot_data_list)))

        ax = sns.boxplot(
            x='Number of models',
            y='Accuracy (%)',
            data=data,
            palette=[colors_dict[model] for model in reversed(models_list)],
            width=.5,
            showfliers=False,
            showcaps=True, linewidth=1.5, linecolor='black',
            zorder=0,
        )

        for day_idx in range(data_list.shape[1]):
            x_jitter = (np.random.rand(data_list.shape[1]) - 0.5) * jitter
            ax.plot(
                [0 + x_jitter[day_idx], 1 + x_jitter[day_idx]], 
                [data_list[0][day_idx], data_list[1][day_idx]], 
                c='gray', alpha=0.5, lw=0.8,
                zorder=1,
            )
            ax.scatter(
                [0 + x_jitter[day_idx], 1 + x_jitter[day_idx]], 
                [data_list[0][day_idx], data_list[1][day_idx]], 
                c='black', s=5, alpha=0.7,
                zorder=2,
            )
        

        ax.plot([0, 0, 1, 1], [102, 104, 104, 102], lw=1, c='black')
        ax.text(0.5, 105, convert_p_val_to_label(p_val_r), ha='center', va='bottom', color='black')

        x_lim = [-.6, 1.6]
        plt.xlim(x_lim)

        y_lim = [0, 130]
        plt.ylim(y_lim)
        y_ticks = [0, 20, 40, 60, 80, 100]
        ax.set_yticks(y_ticks)

        title = f"{models_list[0][-3:]}, Phase {phase}"
        plt.title(title)

        fig_base_path = f"figs/"
        os.makedirs(fig_base_path, exist_ok=True)
        fig_name = f"{data_name}.pdf"

        plt.savefig(fig_base_path + fig_name)

