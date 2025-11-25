import numpy as np
import os
import sys



import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams["font.size"] = 8




print(" ")




length = 15

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
    "MANASNN": '#00a29aa6',
}
colors_list = [colors_dict[model] for model in models_list]


import pickle

data_name = "Fig4b"

with open('data/' + data_name + '.pkl', 'rb') as f:
	r2_data_dict, len_data_dict = pickle.load(f)


# p-value

import pandas as pd
import scipy

mana_models_list = list(filter(lambda model: "MANA" in model, models_list))

pd_data_dict = dict()
p_val_dict = {model: dict() for model in mana_models_list}

for model in models_list:
    pd_data_dict[model] = pd.DataFrame({
        'group': np.repeat(f'train_weeks_{model}', len_data_dict[model]),
        'value': r2_data_dict[model].mean(1),
    })
for mana_model in mana_models_list:
    for model in models_list[:-2]:
        t_stat, p_val = scipy.stats.ttest_rel(pd_data_dict[mana_model]['value'], pd_data_dict[model]['value'],)
        p_val_dict[mana_model][model] = (p_val / 2) if t_stat > 0 else (1 - p_val / 2)

# sns boxplot

import pandas as pd
import seaborn as sns

sns_labels = []
for model_idx, model in enumerate(models_list):
    sns_labels += [f'{models_label_list[model_idx]}'] * len_data_dict[model]
sns_r2s = []
for model in models_list:
    sns_r2s.append(r2_data_dict[model].mean(1))

sns_data = pd.DataFrame({
    "Models": sns_labels,
    "$R^2$": np.concatenate(sns_r2s),
})

f, ax = plt.subplots(figsize=(9, 3.5))
plt.gcf().subplots_adjust(bottom=0.16, top=0.9)

ax = sns.boxplot(
    x=sns_data["Models"], y=sns_data["$R^2$"], palette=colors_list,
    width=.5,
    showcaps=True, linewidth=1.5,
    showfliers=False,
)

ax.set_facecolor('none')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')

plt.ylim(0.2, 1)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])

title = f"Jango, Models summary"
plt.title(title)

fig_base_path = f"figs/"
os.makedirs(fig_base_path, exist_ok=True)
fig_name = f"{data_name}.pdf"

plt.savefig(fig_base_path + fig_name)

