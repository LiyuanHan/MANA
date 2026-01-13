'''
MANA-figures.Fig3bcd:
Because the data required to generate Fig.3b、c、d is too large to upload to GitHub, it is provided via Google Drive instead (https://drive.google.com/drive/folders/1bUp7f4ajYvDBFQuwpgaul8c0koIb-FeN?usp=sharing). Please download the Fig3bcd folder and place it under your_path/MANA-figures/data/, then run: python Fig3bcd.py
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import torch


turn = 0
# testDay = 24 # 1st phase
# testDay = 42 # replace electrode holder
# testDay = 86 # 2nd phase
layer = 7200
figs = {
    '24': 'b',
    '42': 'c',
    '86': 'd'

}
phsae = {'24': '1st phase',
         '42': 'replace phase',
         '86': '2nd phase'}

path_saved_data = './data/Fig3bcd'
filepath_save = './figs'

custom_colors = ["#0094ff", "#008d00", "#ff9200", "#cfb99e", "#e5086a", "#501d8a", "#aa3474", "#ee8c7d"]
custom_cmap =ListedColormap(custom_colors)

def load(path_saved_data, testDay, turn, layer):

    data_raw = torch.load(path_saved_data + f'/raw_data_day_{testDay}.pt')
    data_DR = torch.load(path_saved_data +f'/Before_align_embedding_testDay_{testDay}_gy-p2_.pt')

    data_MA = torch.load(path_saved_data+f'/raw_MA_testDay_{testDay}_turn_{turn}.pt')
    data_DR_MA = torch.load(path_saved_data + f"/align_dir_testDay_{testDay}_turn_{turn}.pt")



    data_AD = torch.load(path_saved_data + f'/AD/layer_{layer}/data_AD_testDay_{testDay}_max_acc.pt',map_location='cpu')
    data_CL_MA_AD = torch.load(path_saved_data + f'/CL_MA_AD/layer_{layer}/data_DR_MA_AD_testDay_{testDay}_max_acc.pt',map_location='cpu')

    label = torch.load(path_saved_data+f"/label_day_{testDay}.pt")



    return data_raw, data_DR, data_MA, data_DR_MA, data_AD, data_CL_MA_AD, label
def set_pane_axis(ax):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])

def PCA_func(X, k):

    U, S, V = torch.pca_lowrank(X, q=k)

    X_reduced = torch.matmul(X, V[:, :k])

    return X_reduced, S
def plot_only_MA(data_visual, type = 't-SNE'):

    feature_dimension = 3
    tsne = TSNE(n_components=feature_dimension, random_state=42)
    raw_data_embedding = tsne.fit_transform(data_visual)

    custom_colors = ["#0094ff", "#008d00", "#ff9200", "#cfb99e", "#e5086a", "#501d8a", "#aa3474", "#ee8c7d"]
    custom_cmap =ListedColormap(custom_colors)

    features_pos = raw_data_embedding
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # features_pos = scaler.fit_transform(features_pos)

    labels_pos = label
    idx1, idx2, idx3 = (0, 1, 2)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"MA only")
    x = ax.scatter(
        features_pos[:, idx1],
        features_pos[:, idx2],
        features_pos[:, idx3],
        cmap=custom_cmap,
        c=labels_pos[:, 0],
        s=0.01,
        vmin=0,
        vmax=7,
    )    

    xc = plt.colorbar(x, fraction=0.03, pad=0.05, ticks=np.linspace(0, 7, 8))
    xc.ax.tick_params(labelsize=15)
    xc.ax.set_title("(cm)", fontsize=10)
    set_pane_axis(ax)
    #set_pane_axis(ax2)
    plt.savefig(filepath_save + f'/Fig3{figs[str(testDay)]}_MA_only_{phsae[str(testDay)]}.pdf')
    plt.show()

def plot_only_DR(data_visual):
    features_pos = data_visual
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # features_pos = scaler.fit_transform(features_pos)

    labels_pos = label
    idx1, idx2, idx3 = (0, 1, 2)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"CL")
    x = ax.scatter(
        features_pos[:, idx1],
        features_pos[:, idx2],
        features_pos[:, idx3],
        cmap=custom_cmap,
        c=labels_pos[:, 0],
        s=0.01,
        vmin=0,
        vmax=7,
    )    

    xc = plt.colorbar(x, fraction=0.03, pad=0.05, ticks=np.linspace(0, 7, 8))
    xc.ax.tick_params(labelsize=15)
    xc.ax.set_title("(cm)", fontsize=10)
    set_pane_axis(ax)
    #set_pane_axis(ax2)
    plt.savefig(filepath_save + f'/Fig3{figs[str(testDay)]}_CL_{phsae[str(testDay)]}.pdf')
    plt.show()

def plot_DR_MA(data_visual):
    for i in range(1): # i can be 0, 1, 2, 3, 4, 5, 6, 7
        features_pos = data_visual[:,i*3:i*3+3]
        labels_pos = label
        idx1, idx2, idx3 = (0, 1, 2)
        fig = plt.figure(figsize=(8, 6))
        ax2 = fig.add_subplot(111, projection="3d")
        ax2.set_title(f"CL + MA")
        x = ax2.scatter(
            features_pos[:, idx1],
            features_pos[:, idx2],
            features_pos[:, idx3],
            cmap=custom_cmap,
            c=labels_pos[:, 0],
            s=0.05,
            vmin=0,
            vmax=7,
        )

        xc = plt.colorbar(x, fraction=0.03, pad=0.05, ticks=np.linspace(0, 7, 8))
        xc.ax.tick_params(labelsize=15)
        xc.ax.set_title("(cm)", fontsize=10)
        set_pane_axis(ax2)
        #set_pane_axis(ax2)
        plt.savefig(filepath_save + f'/Fig3{figs[str(testDay)]}_CL_MA_{phsae[str(testDay)]}_index_{i}.pdf')

    plt.show()
    
def plot_raw_data(data_visual, type = 't-SNE'):

    feature_dimension = 3
    tsne = TSNE(n_components=feature_dimension, random_state=42)
    raw_data_embedding = tsne.fit_transform(data_visual)

    custom_colors = ["#0094ff", "#008d00", "#ff9200", "#cfb99e", "#e5086a", "#501d8a", "#aa3474", "#ee8c7d"]
    custom_cmap =ListedColormap(custom_colors)

    features_pos = raw_data_embedding
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # features_pos = scaler.fit_transform(features_pos)

    labels_pos = label
    idx1, idx2, idx3 = (0, 1, 2)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"RAW")
    x = ax.scatter(
        features_pos[:, idx1],
        features_pos[:, idx2],
        features_pos[:, idx3],
        cmap=custom_cmap,
        c=labels_pos[:, 0],
        s=0.01,
        vmin=0,
        vmax=7,
    )    

    xc = plt.colorbar(x, fraction=0.03, pad=0.05, ticks=np.linspace(0, 7, 8))
    xc.ax.tick_params(labelsize=15)
    xc.ax.set_title("(cm)", fontsize=10)
    set_pane_axis(ax)
    #set_pane_axis(ax2)
    plt.savefig(filepath_save + f'/Fig3{figs[str(testDay)]}_RAW_{phsae[str(testDay)]}.pdf')
    plt.show()

def plot_AD(data_visual, layer = layer, type = 't-SNE'):
    if layer == 7200:
        N, W, H = data_visual[:400, :-25, :-1].shape
        data_visual_trials = data_visual[:400, :-25, :-1].reshape(-1, H)
        labels_pos = data_visual[:400, :-25, -1].reshape(-1 , 1)
    elif layer == 512:
        N, W = data_visual[:400, :512].shape
        data_visual_trials = data_visual[:400, :512]
        labels_pos = data_visual[:, -1]
    elif layer == 8:
        data_visual_trials = data_visual[:, :8]
        labels_pos = data_visual[:, -1]



    feature_dimension = 3
    tsne = TSNE(n_components=feature_dimension, random_state=42)
    raw_data_embedding = tsne.fit_transform(data_visual_trials)

    custom_colors = ["#0094ff", "#008d00", "#ff9200", "#cfb99e", "#e5086a", "#501d8a", "#aa3474", "#ee8c7d"]
    custom_cmap =ListedColormap(custom_colors)

    features_pos = raw_data_embedding
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # features_pos = scaler.fit_transform(features_pos)

    
    idx1, idx2, idx3 = (0, 1, 2)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"AD only")
    x = ax.scatter(
        features_pos[:, idx1],
        features_pos[:, idx2],
        features_pos[:, idx3],
        cmap=custom_cmap,
        c=labels_pos,
        s=0.01,
        vmin=0,
        vmax=7,
    )    

    xc = plt.colorbar(x, fraction=0.03, pad=0.05, ticks=np.linspace(0, 7, 8))
    xc.ax.tick_params(labelsize=15)
    xc.ax.set_title("(cm)", fontsize=10)
    set_pane_axis(ax)
    #set_pane_axis(ax2)
    plt.savefig(filepath_save + f'/Fig3{figs[str(testDay)]}_AD_only_{phsae[str(testDay)]}.pdf')
    plt.show()


def plot_DR_MA_AD_layer(data_visual, layer = layer, type = 't-SNE'):
    if layer == 7200:
        N, W, H = data_visual[:400, :25, :-1].shape
        data_visual_trials = data_visual[:400, :25, :-1].reshape(-1, H)
        labels_pos = data_visual[:400, :25, -1].reshape(-1 , 1)
    elif layer == 512:
        N, W = data_visual[:400, :500].shape
        data_visual_trials = data_visual[:400, :500]
        labels_pos = data_visual[:400, -1]
    
    if type == 'PCA':
        # 先用PCA降维到3维
        feature_dimension = 3
        raw_data_embedding, S = PCA_func(data_visual_trials, k=feature_dimension)
        print('obtaining standard embeddings ...... finished!')
        variance_explained = S**2 / torch.sum(S**2)
        print(f"accumated_var={torch.sum(variance_explained[:feature_dimension])}")
    elif type == 'UMAP':
        feature_dimension = 3
        reducer = umap.UMAP(n_components=feature_dimension)    
        raw_data_embedding = reducer.fit_transform(data_visual_trials)
    elif type == 't-SNE':
        feature_dimension = 3
        tsne = TSNE(n_components=feature_dimension, random_state=42)
        raw_data_embedding = tsne.fit_transform(data_visual_trials)

    custom_colors = ["#0094ff", "#008d00", "#ff9200", "#cfb99e", "#e5086a", "#501d8a", "#aa3474", "#ee8c7d"]
    custom_cmap =ListedColormap(custom_colors)

    features_pos = raw_data_embedding
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # features_pos = scaler.fit_transform(features_pos)

    
    idx1, idx2, idx3 = (0, 1, 2)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"CL + MA + AD")
    x = ax.scatter(
        features_pos[:, idx1],
        features_pos[:, idx2],
        features_pos[:, idx3],
        cmap=custom_cmap,
        c=labels_pos,
        s=0.05,
        vmin=0,
        vmax=7,
    )    

    xc = plt.colorbar(x, fraction=0.03, pad=0.05, ticks=np.linspace(0, 7, 8))
    xc.ax.tick_params(labelsize=15)
    xc.ax.set_title("(cm)", fontsize=10)
    set_pane_axis(ax)
    #set_pane_axis(ax2)
    plt.savefig(filepath_save + f'/Fig3{figs[str(testDay)]}_CL_MA_AD_{phsae[str(testDay)]}.pdf')
    plt.show()




if __name__ == "__main__":
    
    # testDay = 24 # 1st phase
    # testDay = 42 # replace electrode holder
    # testDay = 86 # 2nd phase
    layer = 7200
    turn = 0

    
    ##### ----- Fig3b1~b6 ----- #####
    testDay = 24 # 1st phase
    data_raw, data_DR, data_MA, data_DR_MA, data_AD, data_CL_MA_AD, label = load(path_saved_data, testDay = testDay, turn = turn, layer = layer)

    plot_raw_data(data_raw) # raw data [N*50, 232]
    plot_only_DR(data_DR) # raw data reduced to 3, [N*50, 3]
    plot_DR_MA(data_DR_MA) # raw data using DR + MA, [N*50, 3*8]； alignment with 8 directions
    plot_DR_MA_AD_layer(data_CL_MA_AD, layer = layer)
    plot_only_MA(data_MA) # raw data with alignment [N*50, 232*8]
    plot_AD(data_AD) # raw data using AD, [N*50, 8]

    ##### ----- Fig3c1~c6 ----- #####
    testDay = 42 # replace electrode holder
    data_raw, data_DR, data_MA, data_DR_MA, data_AD, data_CL_MA_AD, label = load(path_saved_data, testDay = testDay, turn = turn, layer = layer)

    plot_raw_data(data_raw) # raw data [N*50, 232]
    plot_only_DR(data_DR) # raw data reduced to 3, [N*50, 3]
    plot_DR_MA(data_DR_MA) # raw data using DR + MA, [N*50, 3*8]； alignment with 8 directions
    plot_DR_MA_AD_layer(data_CL_MA_AD, layer = layer)
    plot_only_MA(data_MA) # raw data with alignment [N*50, 232*8]
    plot_AD(data_AD) # raw data using AD, [N*50, 8]

    ##### ----- Fig3d1~d6 ----- #####
    testDay = 86 # 2nd phase
    data_raw, data_DR, data_MA, data_DR_MA, data_AD, data_CL_MA_AD, label = load(path_saved_data, testDay = testDay, turn = turn, layer = layer)

    plot_raw_data(data_raw) # raw data [N*50, 232]
    plot_only_DR(data_DR) # raw data reduced to 3, [N*50, 3]
    plot_DR_MA(data_DR_MA) # raw data using DR + MA, [N*50, 3*8]； alignment with 8 directions
    plot_DR_MA_AD_layer(data_CL_MA_AD, layer = layer)
    plot_only_MA(data_MA) # raw data with alignment [N*50, 232*8]
    plot_AD(data_AD) # raw data using AD, [N*50, 8]

    plt.close('all')
    
