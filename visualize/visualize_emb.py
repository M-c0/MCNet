import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def visualize_embeddings(features, labels, title="Embedding Visualization", save_path=None, iter=None):

    # 1. 假设 features 是一个 (N, D) 的 tensor，N 是样本数，D 是特征维度
    #    labels 是 (N,) 的 tensor，代表每个样本的类别或模态
    #    如果有模态信息，也可以用不同 marker 或颜色区分
    features = features.cpu().detach().numpy()  # 转为 NumPy 格式用于 sklearn
    # labels = labels.cpu().numpy()               # 标签也转为 NumPy
    labels = np.array(labels)

    # —— 1. 找到出现频次最高的 10 个类别
    cnt = Counter(labels)
    top10 = [lab for lab, _ in cnt.most_common(10)]
    # 只保留这 10 类的样本
    mask = np.isin(labels, top10)
    features = features[mask]
    labels = labels[mask]


    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)  # 映射成 0 ~ n-1

    # 2. 对特征进行标准化（t-SNE 对数据范围较敏感）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 3. 初始化 t-SNE，并降维到 2 维
    tsne = TSNE(n_components=2, perplexity=5, n_iter=2000, init='pca', random_state=42)
    features_tsne = tsne.fit_transform(features_scaled)  # 得到 (N, 2) 的二维坐标

    # 4. 可视化
    plt.figure(figsize=(10, 8))

    # 设置不同颜色的标签，用于可视化（按类或模态）
    num_classes = len(np.unique(labels))
    colors = plt.cm.get_cmap("tab10", 10)

    for i in range(10):
        idx = labels_encoded == i  # 选择标签为 i 的样本
        plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], s=80, alpha=0.9,
                    color=colors(i))  # , label=f'Class {i}')

    # plt.legend()
    plt.title('t-SNE Visualization of Extracted Features')
    # plt.xlabel('Dim 1')
    # plt.ylabel('Dim 2')
    # plt.grid(True)
    # plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path+f'/{iter}',
            dpi=300,
            bbox_inches='tight'  # 剪掉多余空白
        )
        print(f" 已保存可视化图到：{save_path}+'/{iter}'")

    plt.show()
    plt.close()



def visualize_embeddings_3d(
    features,
    labels,
    title="3D t-SNE Visualization",
    save_path=None,
    iter=None
):
    # —— 0. 转 NumPy
    features = features.cpu().detach().numpy()  # (N, D)
    labels   = np.array(labels)                # (N,)

    # —— 1. （可选）只选频次最高的 10 类
    cnt = Counter(labels)
    topK = [lab for lab,_ in cnt.most_common(10)]
    mask = np.isin(labels, topK)
    features = features[mask]
    labels   = labels[mask]

    # —— 2. 编码 & 标准化
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    Xs = StandardScaler().fit_transform(features)

    # —— 3. 3D t-SNE 降到 3 维
    tsne = TSNE(n_components=3, perplexity=5, n_iter=2000,
                init='pca', random_state=42)
    X3 = tsne.fit_transform(Xs)  # (M, 3)

    # —— 4. 作图
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    n_cls = len(le.classes_)
    cmap = plt.cm.get_cmap('tab10', n_cls)

    for i, cls in enumerate(le.classes_):
        idx = (labels_enc == i)
        ax.scatter(
            X3[idx,0], X3[idx,1], X3[idx,2],
            s=50, alpha=0.8,
            color=cmap(i),
            # label=str(cls)
        )
    # 去掉坐标
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(True)
    ax.set_title(title)
    # ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.set_zlabel('Dim 3')
    ax.legend(loc='best', fontsize='small')

    # —— 5. 保存
    if save_path:
        fname = f"{iter}.png" if iter is not None else "tsne3d.png"
        full = os.path.join(save_path, fname)
        plt.savefig(full, dpi=300, bbox_inches='tight')
        print(f"✅ 已保存 3D 可视化图到：{full}")

    plt.show()
    plt.close()

def plot_multi_cmc(cmc_lists, labels, title='CMC Curve Comparison'):
    """
    绘制多条 CMC 曲线并列对比。

    参数
    ----
    cmc_lists : list of list of float
        每个子列表为一条 CMC 曲线的数据，长度相同。
    labels : list of str
        每条曲线对应的标签，用于图例显示。
    title : str
        图表标题。
    """
    ranks = list(range(1, len(cmc_lists[0]) + 1))
    plt.figure(figsize=(6, 4))
    for cmc, label in zip(cmc_lists, labels):
        plt.plot(ranks, cmc, marker='o', linestyle='-', label=label)
    plt.xticks(ranks)
    plt.xlabel('Rank-k')
    plt.ylabel('Recognition Rate')
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

