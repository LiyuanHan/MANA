import numpy as np
import os
import sys



import matplotlib.pyplot as plt
from matplotlib import font_manager
plt.rcParams["font.size"] = 8




print(" ")



models_list = [
    "MANAANN",
    "MANASNN",
]
models_label_dict = {
    "MANAANN": "MANA (ANN)",
    "MANASNN": "MANA (SNN)",
}
models_label_list = [models_label_dict[model] for model in models_list]
colors_dict = {
    "MANAANN": '#da357eb3', 
    "MANASNN": '#00a29aa6'
}
colors_list = [colors_dict[model] for model in models_list]



import pickle

data_name = "Fig4d"

with open('data/' + data_name + '.pkl', 'rb') as f:
	r2_data_dict, len_data_dict = pickle.load(f)

model_shape = r2_data_dict[models_list[0]].shape



# sns boxplot

f, ax = plt.subplots(figsize=(5, 2))

for model_idx, model in enumerate(models_list):
    x_pos = np.array(list(range(len_data_dict[model])))
    ax.errorbar(
        x_pos, r2_data_dict[model].mean(1), yerr=r2_data_dict[model].std(1) / np.sqrt(model_shape[1]),
        color='black', capsize=2, fmt='none', elinewidth=0.5
    )
    ax.plot(
        x_pos, r2_data_dict[model].mean(1),
        marker='s', markersize=3., color=colors_list[model_idx], linewidth=.5,
        label=models_label_list[model_idx],
    )

ax.set_facecolor('none')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')

plt.xlabel('Decoding day')
plt.xticks(list(range(len_data_dict[model])))
plt.ylabel('$R^2$')
plt.ylim(0.8, 1)
plt.yticks([0.8, 0.9, 1.0])

plt.legend(models_label_list, loc='lower right')

title = f"Jango, MANA Daily Summary"
plt.title(title)

fig_base_path = f"figs/"
os.makedirs(fig_base_path, exist_ok=True)
fig_name = f"{data_name}.pdf"

plt.savefig(fig_base_path + fig_name)

