import numpy as np
import os
import sys


import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams["font.size"] = 8


print(" ")


train_weeks_list = [1, 2, 3, 4, 5]
default_train_weeks = 4
other_train_weeks_list = [1, 2, 3, 5]


for model_category in ["MANAANN", "MANASNN"]:

    if "ANN" in model_category:
        data_name = "Fig2b2_1"
    elif "SNN" in model_category:
        data_name = "Fig2b2_2"

    import pickle

    with open('data/' + data_name + '.pkl', 'rb') as f:
        acc_data_dict, len_data_dict = pickle.load(f)


    # p-value calculation

    import pandas as pd
    import scipy
    from scipy import stats

    pd_data_dict = dict()
    p_val_dict = dict()

    for train_weeks in other_train_weeks_list:
        pd_data_dict[train_weeks] = pd.DataFrame({
            'group': np.repeat(f'train_weeks_{train_weeks}', len_data_dict[train_weeks]),
            'value': acc_data_dict[train_weeks].mean(1),
        })

    pd_data_dict[default_train_weeks] = (
        pd.DataFrame({
            'group': np.repeat(f'train_weeks_{default_train_weeks}', len_data_dict[default_train_weeks]),
            'value': acc_data_dict[default_train_weeks].mean(1),
        }),
        pd.DataFrame({
            'group': np.repeat(f'train_weeks_{default_train_weeks}', len_data_dict[other_train_weeks_list[-1]]),
            'value': acc_data_dict[default_train_weeks][1:].mean(1),
        })
    )

    for train_weeks in other_train_weeks_list:
        idx = 0 if train_weeks < default_train_weeks else 1
        t_stat, p_val = stats.ttest_rel(pd_data_dict[default_train_weeks][idx]['value'], pd_data_dict[train_weeks]['value'])
        p_val_dict[train_weeks] = (p_val / 2) if t_stat > 0 else (1 - p_val / 2)



    # sns boxplot

    import seaborn as sns

    sns_labels = []
    for train_weeks in train_weeks_list:
        sns_labels += [f'{train_weeks}'] * len_data_dict[train_weeks]
    sns_accs = []
    for train_weeks in train_weeks_list:
        sns_accs.append(acc_data_dict[train_weeks].mean(1))

    sns_data = pd.DataFrame({
        "Training weeks": sns_labels,
        "Accuracy (%)": np.concatenate(sns_accs),
    })

    if "ANN" in model_category:
        alphaed_palette = ["#e57399", "#e34a7c", "#dc4086", '#ea3988', "#fc2f8b"]
    elif "SNN" in model_category:
        alphaed_palette = ["#55d1cb", "#34cbc3", "#10ab99", '#00a29a', "#03786e"]


    f, ax = plt.subplots(figsize=(5, 3.5))

    ax = sns.boxplot(
        x=sns_data["Training weeks"], y=sns_data["Accuracy (%)"], palette=alphaed_palette,
        width=.5,
        showcaps=True, linewidth=1.5,
        showfliers=False,
    )

    ax.set_facecolor('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')


    def convert_p_val_to_label(pvalue):
        p_val_thres_list = [0.01, 0.05, 0.1]
        for p_val_thres in p_val_thres_list:
            if pvalue < p_val_thres:
                return f"p<{p_val_thres}"
        return "n.s."

    yb = 102
    dy = 9
    y_dict = {1: yb+2*dy, 2: yb+dy, 3: yb, 4: yb, 5: yb}
    dx = 0.03
    h, h_text = 2, 3

    for train_weeks in other_train_weeks_list:
        p_val = p_val_dict[train_weeks]
        x0 = default_train_weeks - 1 + (dx if train_weeks > default_train_weeks else -dx)
        y0 = y_dict[train_weeks]
        x1 = train_weeks - 1
        y1 = y_dict[default_train_weeks]
        x_text = (x0 + x1) / 2
        ax.plot([x0, x0, x1, x1], [y0, y0+h, y0+h, y1], lw=1, c='k')
        ax.text(x_text, y0+h_text, convert_p_val_to_label(p_val), ha='center', va='bottom')

    plt.ylim(0, 130)
    plt.yticks([0, 20, 40, 60, 80, 100, 120], ['0', '20', '40', '60', '80', '100', ''])

    title = f"{model_category}, train weeks comparison"
    plt.title(title)

    fig_base_path = f"figs/"
    os.makedirs(fig_base_path, exist_ok=True)
    fig_name = f"{data_name}.pdf"

    plt.savefig(fig_base_path + fig_name)

