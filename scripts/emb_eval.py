import sys
import os
import numpy as np
import n2v
import matplotlib
import random
from cit_graph import CitGraph
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import label_ranking_average_precision_score
from scipy.stats import spearmanr

def tsne_plot(emb, topics, topic_keywords):
    tsne = TSNE(n_components=2, random_state=0).fit_transform(emb.array)
    matplotlib.rc('font', size=3)
    fig, axes = plt.subplots(nrows=4, ncols=5)
    fig.tight_layout()
    for topic in range(0,20):
        plt.subplot(4, 5, topic+1)
        plt.scatter(tsne[:, 0], tsne[:, 1], s=0.4, linewidths=0, c=topics[:,topic], cmap='Reds')
        plt.title(' '.join(topic_keywords[topic][:4]), pad=-4)
        plt.axis('off')
    plt.savefig('tsne.png', dpi=400, figsize=(35,35))

def plot_precision_curve(pr_at_k):
    plt.plot(range(len(pr_at_k)), pr_at_k)
    plt.xlabel("k")
    plt.ylabel("Precision at k")
    plt.savefig('precision_curve.png')

def plot_MAP_boxplot(map_samples, labels=None):
    print(map_samples)
    plt.ylabel('Mean Average Precision')
    plt.boxplot(map_samples, labels=labels)
    plt.savefig('box.png')

def link_pred_MAP_samples(G, emb, test_edges, model, sample_k, sample_n):
    return [G.MAP_sample(sample_n, emb, E_obs=test_edges, model=model)
            for _ in range(sample_k)]

def link_pred_pr_at_k_samples(G, emb, test_edges, sample_k, sample_n):
    return [CG.pr_at_k_sample(100, sample_n, emb, E_obs=test_edges, model=reg)
            for _ in range(sample_k)]

def train_reg_model(G, emb, train_edges):
    X_true = [emb.to_hadamard_vec(edge) for edge in train_edges]
    X_false = [emb.to_hadamard_vec(edge) for edge in G.negative_edges(len(train_edges))]
    X = np.concatenate((X_true, X_false))
    y = np.concatenate((np.ones((len(X_true))), np.zeros(len(X_false))))
    return LinearRegression().fit(X, y)

def pr_curve(G, v_i, emb, model):
    neighbors = set(G.neighbors(v_i))
    y_true = np.array([int(v_j in neighbors)
        for v_j in [emb.vector_idx_to_node[j] for j in range(len(emb.array))]])
    y_scores = np.array(model.predict(emb.array * emb.to_vec(v_i)))
    return precision_recall_curve(y_true, y_scores)

def plot_pr_curve(recall, precision):
    plt.step(recall, precision, color='b', alpha=0.2,
            where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('pr_curve.png')

def AP(G, v_i, emb, model, test_edges=None):
    if test_edges == None:
        test_nodes = set(G.nodes())
        true_neighbors = set(G.neighbors(v_i))
    else:
        # For link prediction, only include test edges and true negative edges
        true_neighbors = set([node for edge in test_edges if v_i in edge for node in edge])
        test_nodes = set(G.nodes()).difference(set(G.neighbors(v_i))).union(true_neighbors)

    y_true = np.array([int(v_j in true_neighbors)
        for v_j in test_nodes])
    scores = np.array(model.predict(emb.array * emb.to_vec(v_i)))
    y_scores = [scores[emb.node_to_vector_idx[v_j]] for v_j in test_nodes]

    return average_precision_score(y_true, y_scores)

def mAP_sample(G, emb, model, E_obs, k, n):
    if E_obs == None:
        test_edges = list(G.edges())
    else:
        test_edges = E_obs
    nodes = [node for edge in test_edges for node in edge]
    return np.array([np.mean([AP(G, node, emb, model, E_obs)
        for node in random.sample(nodes, n)]) for _ in range(k)])

def train_multilabel_model(G, emb, nodes, label_dict):
    mlb = MultiLabelBinarizer()
    X = np.array([emb.to_vec(node) for node in nodes])
    y = mlb.fit_transform(np.array([label_dict[node] for node in nodes]))
    model = OneVsRestClassifier(SVC(kernel='linear'))
    model.fit(X, y)
    return (model, mlb)

def load_label_data(node_data_path):
    with open(node_data_path) as f:
        data = [line.split('\t') for line in f.readlines()]
    return dict([(datum[0], datum[-1].split()) for datum in data])

def run_label_ranking(G):
    nodes = list(set([node for edge in G.edges() for node in edge]))
    print("outer node count:")
    print(len(nodes))
    emb = G.embed_edges(list(G.edges()), d=24, use_cache=False)
    embedded_set = set(emb.node_to_vector_idx.keys())

    train_nodes = list(set([node for edge in train_edges for node in edge if node in embedded_set]))
    test_nodes = list(set([node for edge in test_edges for node in edge if node in embedded_set]))

    label_data = load_label_data(node_path)
    model, mlb = train_multilabel_model(G, emb, train_nodes, label_data)
    y_score = model.predict([emb.to_vec(node) for node in test_nodes])
    y_true = mlb.transform(np.array([label_data[node] for node in test_nodes]))
    print(label_ranking_average_precision_score(y_true, y_score))

def load_lin_mesh(path):
    with open(path) as f:
        return dict((((x[0], x[1]), float(x[2]))
            for x in map(lambda x:x.split(), f.readlines())))

def run_spearmanr(mesh_dict, emb, reg_model, test_edges):
    predictions = reg_model.predict
    a = reg_model.predict(emb.to_hadamard_vec(mesh_dict.keys()))
    b = np.array(mesh_dict.values())
    print(a.shape)
    plt.plot(a)
    plt.savefig('spearmanr.png')
    print(spearmanr(a, b))

#G = CitGraph()
#G.load_edges(edge_path)
#
#train_edges, test_edges = G.train_test_split_edges()
#
#emb = G.embed_edges(train_edges, d=24, use_cache=False)
#reg_model = train_reg_model(G, emb, train_edges)
#
#lin_scores = load_lin_mesh('sim_mesh_lin.csv')
#
#run_spearmanr(lin_scores, emb, reg_model, test_edges)

#reg_pr_at_k = link_pred_pr_at_k_samples(G, emb, test_edges, model=reg_model, 1024, 5)

#cos_MAP = link_pred_MAP_samples(G, emb, test_edges, 'cos', 5, 1024)

#plot_MAP_boxplot(np.array(reg_mAPs).T, labels=dimensions)
#cos_pr_at_k = link_pred_pr_at_k_samples(G, emb, test_edges, model='cos', 1024, 5)

#topics = G.load_lda_topics(lda_topic_composition_path)
#topic_keywords = G.load_topic_keywords(lda_topics_path)
