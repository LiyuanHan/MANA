
import numpy as np


model_list = ['aAc', 'aSc', 'pAc', 'pSc', 'wAc', 'wSc']

color_dict = {
    'aAc': '#da357e', 
    'aSc': '#31aca6',
    'pAc': '#da6d34',
    'pSc': '#3074ab',
    'wAc': '#d63f2e',
    'wSc': '#075f55',
}

label_dict = {
    'aAc': 'CL+MA+AD, crossday', 
    'aSc': 'CL+MA+AD SNN, crossday',
    'pAc': 'CL+AD, crossday',
    'pSc': 'CL+AD SNN, crossday',
    'wAc': 'AD, crossday',
    'wSc': 'AD, crossday',
}




import pickle


with open('./data/Fig3ej.pkl', 'rb') as f:
	acc_last_dict = pickle.load(f)




for model in model_list:

    data = acc_last_dict[model].mean(1)

    # print(model)
    # print("mean:    ", data.mean())
    # print("median:  ", np.median(data))
    # print("std:     ", data.std())
    # print(" ")


def convert_p_val_to_label(pvalue):
    if pvalue < 0.01:
        return "p<0.01"



import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt

width = .5

x1, x2, x3 = 0, 1, 2
y, h, dh = 102, 1, 4

y_lim = [45, 115]

#

sns.set_theme(style="white")

len_day = acc_last_dict['aAc'].shape[0]
pd_wAc = pd.DataFrame({'Algorithms': np.repeat('AD', len_day), 'Accuracy (%)': np.stack(acc_last_dict['wAc']).mean(1)})
pd_pAc = pd.DataFrame({'Algorithms': np.repeat('CL+AD', len_day), 'Accuracy (%)': np.stack(acc_last_dict['pAc']).mean(1)})
pd_aAc = pd.DataFrame({'Algorithms': np.repeat('CL+MA+AD', len_day), 'Accuracy (%)': np.stack(acc_last_dict['aAc']).mean(1)})
df = pd.concat([pd_wAc, pd_pAc, pd_aAc])


_, p_val_1 = scipy.stats.ttest_ind(pd_pAc['Accuracy (%)'], pd_aAc['Accuracy (%)'], equal_var=False)
# print("CL+MA+AD vs CL+AD p_val", p_val_1)
_, p_val_2 = scipy.stats.ttest_ind(pd_wAc['Accuracy (%)'], pd_aAc['Accuracy (%)'], equal_var=False)
# print("CL+MA+AD vs AD p_val", p_val_2)

f, ax = plt.subplots()

ax = sns.boxplot(
    data=df, x='Algorithms', y='Accuracy (%)', hue='Algorithms', 
    width=width,
    palette=[color_dict['wAc'], color_dict['pAc'], color_dict['aAc']], 
    showcaps=True, linewidth=1.5,
)

ax.plot([x1, x1, x3, x3], [y+dh, y+h+dh, y+h+dh, y+dh], lw=1, c='k')
ax.text((x1 + x3) / 2, y + h + dh, convert_p_val_to_label(p_val_2), ha='center', va='bottom')
ax.plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1, c='k')
ax.text((x2 + x3) / 2, y + h, convert_p_val_to_label(p_val_1), ha='center', va='bottom')

plt.ylim(y_lim)

path__savefig = './figs/Fig3e.pdf'
plt.savefig(path__savefig, format='pdf')
plt.close()
print(f"image saved to {path__savefig}")






sns.set_theme(style="white")

len_day = acc_last_dict['aSc'].shape[0]
pd_wSc = pd.DataFrame({'Algorithms': np.repeat('AD', len_day), 'Accuracy (%)': np.stack(acc_last_dict['wSc']).mean(1)})
pd_pSc = pd.DataFrame({'Algorithms': np.repeat('CL+AD', len_day), 'Accuracy (%)': np.stack(acc_last_dict['pSc']).mean(1)})
pd_aSc = pd.DataFrame({'Algorithms': np.repeat('CL+MA+AD', len_day), 'Accuracy (%)': np.stack(acc_last_dict['aSc']).mean(1)})
df = pd.concat([pd_wSc, pd_pSc, pd_aSc])

_, p_val_1 = scipy.stats.ttest_ind(pd_pSc['Accuracy (%)'], pd_aSc['Accuracy (%)'], equal_var=False)
# print("SNN - CL+MA+AD vs CL+AD p_val", p_val_1)
_, p_val_2 = scipy.stats.ttest_ind(pd_wSc['Accuracy (%)'], pd_aSc['Accuracy (%)'], equal_var=False)
# print("SNN - CL+MA+AD vs AD p_val", p_val_2)

f, ax = plt.subplots()

ax = sns.boxplot(
    data=df, x='Algorithms', y='Accuracy (%)', hue='Algorithms', 
    width=width,
    palette=[color_dict['wSc'],color_dict['pSc'], color_dict['aSc']], 
    showcaps=True, linewidth=1.5,
)

ax.plot([x1, x1, x3, x3], [y+dh, y+h+dh, y+h+dh, y+dh], lw=1, c='k')
ax.text((x1 + x3) / 2, y + h + dh, convert_p_val_to_label(p_val_2), ha='center', va='bottom')
ax.plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1, c='k')
ax.text((x2 + x3) / 2, y + h, convert_p_val_to_label(p_val_1), ha='center', va='bottom')

plt.ylim(y_lim)

path__savefig = './figs/Fig3j.pdf'
plt.savefig(path__savefig, format='pdf')
plt.close()
print(f"image saved to {path__savefig}")

