import sys
import os
import numpy as np
import matplotlib
from mpl_toolkits import mplot3d
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import jsonlines

def get_counts(examples, key_field):
    counts = defaultdict(int)
    for ex in examples:
        item = ex[key_field]
        if item:
            counts[item] += 1
    counts = list(counts.items())
    counts.sort(key=lambda x: -x[1])
    return counts

def tsne_plot(labels, vectors, out_path, rows, cols):
    tsne = TSNE(n_components=2, random_state=0).fit_transform(vectors)
    matplotlib.rc('font', size=7)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, dpi=400, figsize=(cols*2,rows*2))
    for i, label in enumerate(sorted(list(set(labels)))):
        c = np.asarray([(0.8, 0.1, 0.2, 0.8) if l==label else (0.6,0.6,0.6,0.2) for l in labels])
        plt.subplot(rows, cols, i+1)
        plt.scatter(tsne[:, 0], tsne[:, 1], s=1, c=c, linewidths=0)
        plt.title(label, pad=-4)
        plt.axis('off')
    fig.tight_layout()
    plt.savefig(out_path)

def tsne_plot_3d(labels, vectors, dates, out_path, rows, cols):
    tsne = TSNE(n_components=2, random_state=0).fit_transform(vectors)
    matplotlib.rc('font', size=3)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, dpi=400, figsize=(cols*2,rows*2))
    for i, label in enumerate(set(labels)):
        c = np.asarray([(0.8, 0.1, 0.2, 0.8) if l==label else (0.6,0.6,0.6,0.2) for l in labels])
        plt.subplot(cols, rows, i+1, projection='3d')
        plt.scatter(tsne[:, 0], tsne[:, 1], s=1, c=c, linewidths=0)
        plt.title(label, pad=-4)
    fig.tight_layout()
    plt.savefig(out_path)

def get_emb_lookup(emb_path):
    emb_lookup = {}
    with open(emb_path) as f:
        for line in f.readlines():
            items = line.split()
            emb_lookup[items[0]] = np.array(list(map(float, items[1:])))
    return emb_lookup

if __name__ == "__main__":
    emb_path = sys.argv[1]
    data_path = sys.argv[2]
    top_n = int(sys.argv[3])
    key_field = sys.argv[4]
    out_path = sys.argv[5]
    cols = int(sys.argv[6])
    rows = int(sys.argv[7])

    examples = list(jsonlines.open(data_path))

    counts = get_counts(examples, key_field)
    top_items = set([w[0] for w in counts[:top_n]])

    emb_lookup = get_emb_lookup(emb_path)
    examples = [example for example in examples if example[key_field] in top_items and example['paper_id'] in emb_lookup.keys()]
    labels = [example[key_field] for example in examples]
    emb_matrix = [emb_lookup[example['paper_id']] for example in examples]

    tsne_plot(labels, emb_matrix, out_path, rows, cols)
    #tsne_plot_3d(labels, emb_matrix, dates, out_path)
