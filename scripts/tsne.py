import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def tsne_plot(labels, vectors, out_path):
    tsne = TSNE(n_components=2, random_state=0).fit_transform(vectors)
    matplotlib.rc('font', size=3)
    fig, axes = plt.subplots(nrows=2, ncols=5, dpi=300, figsize=(7.5,3))
    for i, label in enumerate(set(labels)):
        c = [7-float(l == label)*3 for l in labels]
        plt.subplot(2, 5, i+1)
        plt.scatter(tsne[:, 0], tsne[:, 1], s=1, c=c, linewidths=0, cmap='tab20c')
        plt.title(label, pad=-4)
        plt.axis('off')
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
    out_path = sys.argv[3]
    emb_lookup = get_emb_lookup(emb_path)
    workshop_lookup = get_workshop_lookup(workshop_path)

    top_workshops = [
        "Machine Translation",
        "Discourse and Dialogue",
        "Chinese Language Processing",
        "CoNLL",
        "Proceedings of the SIGDIAL Conference",
        "Linguistic Annotation Workshop",
        "Innovative Use of NLP for Building Educational Applications",
        "NLG",
        "Proceedings of BioNLP Workshop",
        "International Conference on Computational Semantics (IWCS)"
    ]

    emb_lookup = {k:v for k,v in emb_lookup.items() if workshop_lookup.get(k[:6]) and workshop_lookup[k[:6]] in top_workshops}

    vocabulary = {
        "workshop_labels": {label: idx for idx, label in enumerate(set(workshop_lookup.values()))},
        "workshop_indices": dict(enumerate(set(workshop_lookup.values()))),
        "paper_labels": {label: idx for idx, label in enumerate(workshop_lookup.keys())},
        "paper_indices": dict(enumerate(workshop_lookup.keys()))
    }

    emb_matrix = np.array(list(emb_lookup.values()))
    labels = [workshop_lookup[paper_id[:6]]
        for paper_id in emb_lookup.keys()]
    tsne_plot(labels, emb_matrix, out_path)
