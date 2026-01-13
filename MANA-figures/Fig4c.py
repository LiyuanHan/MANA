import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import torch
import pickle


path_saved_data = './data/Fig3bcd'
filepath_save = './figs'


def plot_neural_model(data_Jange_aligned, label):
    feature_dimension = 2
    tsne = TSNE(n_components=feature_dimension, random_state=42)
    raw_data_embedding = tsne.fit_transform(data_Jange_aligned)

    # 自定义颜色
    custom_colors = ["#0094ff", "#008d00", "#ff9200", "#cfb99e", 
                     "#e5086a", "#501d8a", "#aa3474", "#ee8c7d"]
    custom_cmap = ListedColormap(custom_colors)

    features_pos = raw_data_embedding
    labels_pos = np.array(label).astype(int)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        features_pos[:, 0],
        features_pos[:, 1],
        c=labels_pos,
        cmap=custom_cmap,
        s=5,
        vmin=0,
        vmax=7,
        alpha=0.8
    )

    # ax.set_title("Neural Model Aligned (2D t-SNE)", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # 颜色条
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.05, ticks=np.linspace(0, 7, 8))
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_title("Direction", fontsize=10)

    plt.tight_layout()
    # plt.savefig(filepath_save + f'/MANA-SNN_Jango_test_6_2d_t-SNE_(scatter).pdf')
    plt.show()

def plot_neural_model_line(data_Jange_aligned, label):
    feature_dimension = 2
    # tsne = TSNE(n_components=feature_dimension, random_state=42)
    tsne = TSNE(
    n_components=feature_dimension,
    random_state=42,
    init="pca",              # 固定初始化：pca 更稳定（也可用 "random"，但两边必须一致）
    learning_rate=200.0,     # 不要用默认 'auto'
    perplexity=30.0,
    early_exaggeration=12.0,
    n_iter=2000,             # sklearn新版本叫 max_iter；老版本叫 n_iter
    metric="euclidean",
    method="barnes_hut",     # 或 "exact"（更稳但更慢）
    angle=0.5,
    # square_distances=True,   # 老版本可能没有；如果报错就删掉这一行
    verbose=0
)
    raw_data_embedding = tsne.fit_transform(data_Jange_aligned)

    # 自定义颜色
    custom_colors = ["#0094ff", "#008d00", "#ff9200", "#cfb99e", 
                     "#e5086a", "#501d8a", "#aa3474", "#ee8c7d"]
    custom_cmap = ListedColormap(custom_colors)

    features_pos = raw_data_embedding
    labels_pos = np.array(label).squeeze().astype(int)  # ✅ 修复关键

    fig, ax = plt.subplots(figsize=(8, 6))

    # 获取所有类别
    unique_labels = np.unique(labels_pos)
    n_dirs = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, n_dirs))

    # 按方向画线
    for i, d in enumerate(unique_labels):
        mask = labels_pos == d
        traj = features_pos[mask]
        ax.plot(
            traj[:, 0], traj[:, 1],
            color=custom_colors[d % len(custom_colors)],
            linewidth=1.0,
            alpha=0.8,
            label=f'Dir {d}'
        )

    # ax.set_title("Neural Model Aligned (2D t-SNE trajectories)", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Direction", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    # plt.savefig(filepath_save + f'/MANA-SNN_Jango_test_6_2d_t-SNE_(line).pdf')
    plt.xlim(-100,100)
    plt.ylim(-100,100)

if __name__ == "__main__":
    
    ##### ----- Fig4c (MANA-ANN) in the revised manuscript ----- #####
    # with open('./data/MANAANN_Jango-test_6.pkl', 'rb') as f:
    #     data_J = pickle.load(f)
    #     print('load standard embeddings ...... finished!')
    data_J = torch.load('./data/MANAANN_Jango-test_6.pt', map_location='cpu')
    domain_idx = 0
    N, T, C = data_J[domain_idx][0].shape
    timestemps = 14

    data_Jange_aligned = data_J[domain_idx][0][:,timestemps:,:].numpy().reshape(-1, C)
    discrete_label = data_J[domain_idx][1][:,timestemps:,:].numpy().reshape(-1, 1)


    # plot_neural_model(data_Jange_aligned, discrete_label)
    plot_neural_model_line(data_Jange_aligned, discrete_label)
    # plt.savefig('./figs/Fig4c-MANA-ANN.pdf')
    plt.show()

    

    # ##### ----- Fig4c (MANA-SNN) CL+MA+AD for Jango as MANA-SNN in the revised manuscript ----- #####
    # with open('./data/MANASNN_Jango-test_6.pkl', 'rb') as f:
    #     data_J = pickle.load(f)
    data_J = torch.load('./data/MANASNN_Jango-test_6.pt', map_location='cpu')
    timestemps = 14
    data_Jange_aligned = data_J['predicted_positions'].data.cpu().numpy()
    N, T, C = data_Jange_aligned.shape
    data_Jange_aligned = data_Jange_aligned[:, timestemps:, :].reshape(-1, C)
    discrete_label = np.repeat(data_J['real_labels'].data.cpu().numpy(),30-timestemps).reshape(-1,1)
    # plot_neural_model(data_Jange_aligned, discrete_label)
    plot_neural_model_line(data_Jange_aligned, discrete_label)
    # plt.savefig('./figs/Fig4c-MANA-SNN.pdf')
    plt.show()

    plt.close('all')
    
