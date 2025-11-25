##

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

torch.backends.cudnn.benchmark = True

import pickle


with open('./data/Fig3gl.pkl', 'rb') as f:
    domain_weight_models = pickle.load(f)



for i, model in enumerate(["ANN", "SNN"]):

    if model == "ANN":
        colors = ['#da357e', '#4B5D67', '#3E4E50', '#5C5366', '#4A4843']
        tag = 'g'
    else:
        colors = ['#31aca6', '#4B5D67', '#3E4E50', '#5C5366', '#4A4843']
        tag = 'l'

    domain_weight_all = domain_weight_models[i]

    mean_data = np.mean(domain_weight_all, axis=0)
    std_data = np.std(domain_weight_all, axis=0)

    plt.figure()

    x = np.arange(150)
    labels = ['New NN', 'Domain NN 1', 'Domain NN 2', 'Domain NN 3', 'Domain NN 4']
    for i in range(5):
        plt.plot(x, mean_data[i, :], c=colors[i])
        plt.fill_between(x, mean_data[i, :] - std_data[i, :], mean_data[i, :] + std_data[i, :], facecolor=colors[i], alpha=0.2, label=labels[i])

        plt.legend()
        plt.ylim([0.05, 0.4])

    plt.xlabel('Epochs')
    plt.ylabel('Importance')
        
    path__savefig = f'./figs/Fig3{tag}.pdf'
    plt.savefig(path__savefig, format='pdf')
    plt.close()
    print(f"image saved to {path__savefig}")



