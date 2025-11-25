import os

import numpy as np
import matplotlib.pyplot as plt
import torch

torch.backends.cudnn.benchmark = True

mixlen_list = [8, 16, 24, 26, 28]

acc_mean__ANN = []
acc_median__ANN = []
acc_std__ANN = []
acc_mean__SNN = []
acc_median__SNN = []
acc_std__SNN = []


mixlen_total = len(mixlen_list)

import pickle

with open('./data/Fig2b1.pkl', 'rb') as f:
	acc__ANN, acc__SNN = pickle.load(f)


for i, _ in enumerate(mixlen_list):

	acc__ANN_of_mixlen = acc__ANN[i]
	acc__SNN_of_mixlen = acc__SNN[i]

	stat_over_variables_of_acc__ANN_of_mixlen = acc__ANN_of_mixlen.mean(1) 
	stat_over_variables_of_acc__SNN_of_mixlen = acc__SNN_of_mixlen.mean(1)

	mean_over_seeds_of_stat_over_variables_of_acc__ANN_of_mixlen = stat_over_variables_of_acc__ANN_of_mixlen.mean()
	median_over_seeds_of_stat_over_variables_of_acc__ANN_of_mixlen = np.median(stat_over_variables_of_acc__ANN_of_mixlen)
	std_over_seeds_of_stat_over_variables_of_acc__ANN_of_mixlen = stat_over_variables_of_acc__ANN_of_mixlen.std()
	mean_over_seeds_of_stat_over_variables_of_acc__SNN_of_mixlen = stat_over_variables_of_acc__SNN_of_mixlen.mean()
	median_over_seeds_of_stat_over_variables_of_acc__SNN_of_mixlen = np.median(stat_over_variables_of_acc__SNN_of_mixlen)
	std_over_seeds_of_stat_over_variables_of_acc__SNN_of_mixlen = stat_over_variables_of_acc__SNN_of_mixlen.std()

	acc_mean__ANN.append(mean_over_seeds_of_stat_over_variables_of_acc__ANN_of_mixlen)
	acc_median__ANN.append(median_over_seeds_of_stat_over_variables_of_acc__ANN_of_mixlen)
	acc_std__ANN.append(std_over_seeds_of_stat_over_variables_of_acc__ANN_of_mixlen)
	acc_mean__SNN.append(mean_over_seeds_of_stat_over_variables_of_acc__SNN_of_mixlen)
	acc_median__SNN.append(median_over_seeds_of_stat_over_variables_of_acc__SNN_of_mixlen)
	acc_std__SNN.append(std_over_seeds_of_stat_over_variables_of_acc__SNN_of_mixlen)


# print("ANN:", acc_median__ANN)
# print("SNN:", acc_median__SNN)
# print(" ")


fig, ax = plt.subplots()
x_trials_A = [i + 0.8 for i in range(mixlen_total)]
x_trials_S = [i + 1.2 for i in range(mixlen_total)]

plt.errorbar(
	x_trials_A, acc_mean__ANN, yerr=acc_std__ANN,
	capsize=5, marker='s', fmt='none', elinewidth=0.5, ecolor='#000000', alpha=0.8
)

plt.errorbar(
	x_trials_S, acc_mean__SNN, yerr=acc_std__SNN,
	capsize=5, marker='s', fmt='none', elinewidth=0.5, ecolor='#000000', alpha=0.8
)

plt.bar(
	x_trials_A, acc_median__ANN,
	capsize=5, width=0.3, linewidth=0.5, linestyle='solid', color='#e5086a', alpha=0.8,
)

plt.bar(
	x_trials_S, acc_median__SNN,
	capsize=5, width=0.3, linewidth=0.5, linestyle='solid', color='#00a29a', alpha=0.8,
)

plt.xticks([1, 2, 3, 4, 5], [8, 16, 24, 26, 28])
plt.xlabel('Trials/day for fine-tuning')
plt.ylabel('Accuracy (%)')

path__savefig = './figs/Fig2b1.pdf'
plt.savefig(path__savefig, format='pdf')


