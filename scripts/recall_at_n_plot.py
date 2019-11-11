import sys
import numpy as np
import matplotlib.pyplot as plt

def plot(datasets):
    fig, ax = plt.subplots()
    labels = []
    for i, (k, (X, Y)) in enumerate(datasets.items()):
        labels.append(k)
        ax.plot(X, Y)
    plt.legend(labels,loc='upper right')
    ax.set_xlabel("N")
    ax.set_ylabel("recall@N")
    plt.savefig('recall_plot.png')

if __name__ == "__main__":
    ngram_metrics_path = sys.argv[1]
    unigram_metrics_path = sys.argv[2]
    datasets = {}
    with open(ngram_metrics_path) as f:
        datasets['trigram'] = zip(*enumerate([float(line.split()[-1]) for line in f.readlines()]))
    with open(unigram_metrics_path) as f:
        datasets['unigram'] = zip(*enumerate([float(line.split()[-1]) for line in f.readlines()]))
    plot(datasets)
