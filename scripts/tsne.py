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

def get_workshop_counts(file_path, workshop_lookup):
    workshop_counts = defaultdict(int)
    for ex in jsonlines.open(file_path):
        workshop = workshop_lookup.get(ex['paper_id'][:6])
        if workshop:
            workshop_counts[workshop] += 1
    workshop_counts = list(workshop_counts.items())
    workshop_counts.sort(key=lambda x: -x[1])
    return workshop_counts

def tsne_plot(labels, vectors, out_path):
    tsne = TSNE(n_components=2, random_state=0).fit_transform(vectors)
    matplotlib.rc('font', size=3)
    fig, axes = plt.subplots(nrows=4, ncols=6, dpi=400, figsize=(12,8))
    for i, label in enumerate(set(labels)):
        c = np.asarray([(0.8, 0.1, 0.2, 0.8) if l==label else (0.6,0.6,0.6,0.2) for l in labels])
        plt.subplot(4, 6, i+1)
        plt.scatter(tsne[:, 0], tsne[:, 1], s=1, c=c, linewidths=0)
        plt.title(label, pad=-4)
        plt.axis('off')
    fig.tight_layout()
    plt.savefig(out_path)

def tsne_plot_3d(labels, vectors, dates, out_path):
    tsne = TSNE(n_components=2, random_state=0).fit_transform(vectors)
    matplotlib.rc('font', size=3)
    fig, axes = plt.subplots(nrows=2, ncols=5, dpi=400, figsize=(10,4))
    for i, label in enumerate(set(labels)):
        c = np.asarray([(0.8, 0.1, 0.2, 0.8) if l==label else (0.6,0.6,0.6,0.2) for l in labels])
        plt.subplot(2, 5, i+1, projection='3d')
        plt.scatter(tsne[:, 0], tsne[:, 1], s=1, c=c, linewidths=0)
        plt.title(label, pad=-4)
    fig.tight_layout()
    plt.savefig(out_path)

def get_workshop_lookup(workshop_path):
    workshop_lookup = {}
    with open(workshop_path) as f:
        for line in f.readlines():
            items = line.split('\t')
            for paper_id in items[2:]:
                workshop_lookup[paper_id] = items[1]
    return workshop_lookup

def get_date_lookup(date_path):
    date_lookup = {}
    with open(date_path) as f:
        for line in f.readlines():
            items = line.split('\t')
            date_lookup[items[0]] = items[1]
    return date_lookup

def get_emb_lookup(emb_path):
    emb_lookup = {}
    with open(emb_path) as f:
        for line in f.readlines():
            items = line.split()
            emb_lookup[items[0]] = np.array(list(map(float, items[1:])))
    return emb_lookup

if __name__ == "__main__":
    emb_path = sys.argv[1]
    workshop_path = sys.argv[2]
    date_path = sys.argv[3]
    out_path = sys.argv[4]
    emb_lookup = get_emb_lookup(emb_path)
    workshop_lookup = get_workshop_lookup(workshop_path)
    date_lookup = get_date_lookup(date_path)

    workshop_counts = get_workshop_counts('/shared-1/projects/concite/data/acl_n2v.jsonl', workshop_lookup)

    top_workshops = [w[0] for w in workshop_counts[:10]]

    emb_lookup = {k:v for k,v in emb_lookup.items() if workshop_lookup.get(k[:6]) and workshop_lookup[k[:6]] in top_workshops and k in date_lookup.keys()}

    vocabulary = {
        "workshop_labels": {label: idx for idx, label in enumerate(set(workshop_lookup.values()))},
        "workshop_indices": dict(enumerate(set(workshop_lookup.values()))),
        "paper_labels": {label: idx for idx, label in enumerate(workshop_lookup.keys())},
        "paper_indices": dict(enumerate(workshop_lookup.keys()))
    }

    emb_matrix = np.array(list(emb_lookup.values()))
    labels = [workshop_lookup[paper_id[:6]]
        for paper_id in emb_lookup.keys()]
    dates = [date_lookup.get(paper_id, -1)
        for paper_id in emb_lookup.keys()]
    #tsne_plot(labels, emb_matrix, out_path)
    tsne_plot_3d(labels, emb_matrix, dates, out_path)
