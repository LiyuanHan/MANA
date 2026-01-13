

import matplotlib.pyplot as plt


model_list = ['aAc', 'aSc']


val_range = 0.0015

test_Day = 37


import pickle


with open('./data/Fig3hm.pkl', 'rb') as f:
    model_params_dict = pickle.load(f)

for model in model_list:

    if model == "aAc":
        tag = 'h'
    else:
        tag = 'm'

    weights = model_params_dict[model].mean(axis=1).reshape(50, 144)


    import matplotlib.pyplot as plt

    plt.figure(dpi=300)
    im = plt.imshow(weights, cmap='seismic', interpolation='nearest')
    plt.colorbar()
    im.set_clim(-val_range, val_range)

    path__savefig = f'./figs/Fig3{tag}.pdf'
    plt.savefig(path__savefig, format='pdf')
    plt.close()
    print(f"image saved to {path__savefig}")



