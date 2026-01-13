
import numpy as np


np.set_printoptions(threshold=np.inf)

seeds = [0, 1, 2, 5, 7]

model_list = ['aAc', 'aSc', 'pAc', 'pSc', 'wAc', 'wSc']

color_dict = {
    'aAc': '#da357e', 
    'aSc': '#31aca6',
    'pAc': '#da6d34',    # #e29135
    'pSc': '#3074ab',    # #4a5f7e
    'wAc': '#d63f2e',
    'wSc': '#075f55',
}
label_dict = {
    'aAc': 'CL+MA+AD', 
    'aSc': 'CL+MA+AD SNN',
    'pAc': 'CL+AD',
    'pSc': 'CL+AD SNN',
    'wAc': 'AD',
    'wSc': 'AD',
}


import pickle


with open('./data/Fig3fk.pkl', 'rb') as f:
	loss_turn_test_dict = pickle.load(f)




for model in model_list:

    mean = np.average(loss_turn_test_dict[model], axis=0)
    std = loss_turn_test_dict[model].std(axis=0)



import matplotlib.pyplot as plt

x_ax = np.array([i for i in range(1, 151)])
x_ticks = [0, 50, 100, 150]


fig, ax = plt.subplots(1, 1, figsize=(5, 4))
fig_models = ['aAc', 'pAc', 'wAc']
y_lim = [-0.2, 3.2]


for model in fig_models:

    mean = np.average(loss_turn_test_dict[model], axis=0)
    std = loss_turn_test_dict[model].std(axis=0)

    ax.fill_between(x_ax, mean - std, mean + std, facecolor=color_dict[model], alpha=0.3)
    ax.plot(x_ax, mean, c=color_dict[model], label=label_dict[model])

plt.xlabel('Epochs')
plt.ylabel('Loss')
ax.xaxis.set_ticks(x_ticks)
ax.set_ylim(y_lim)
ax.legend()


path__savefig = './figs/Fig3f.pdf'
plt.savefig(path__savefig, format='pdf')
plt.close()
print(f"image saved to {path__savefig}")




fig, ax = plt.subplots(1, 1, figsize=(5, 4))
fig_models = ['aSc', 'pSc', 'wSc']


for model in fig_models:

    mean = np.average(loss_turn_test_dict[model], axis=0)
    std = loss_turn_test_dict[model].std(axis=0)

    ax.fill_between(x_ax, mean - std, mean + std, facecolor=color_dict[model], alpha=0.3)
    ax.plot(x_ax, mean, c=color_dict[model], label=label_dict[model])

plt.xlabel('Epochs')
plt.ylabel('Loss')
ax.xaxis.set_ticks(x_ticks)
ax.set_ylim(y_lim)
ax.legend()


path__savefig = './figs/Fig3k.pdf'
plt.savefig(path__savefig, format='pdf')
plt.close()
print(f"image saved to {path__savefig}")

