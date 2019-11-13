import sys
import os
import numpy as np
import json
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

def load_data(path):
    with open(path) as f:
        if path.split('.')[-1] == 'json':
            data_dict = json.load(f)
            return zip(*[(i, data_dict["recall_at_{}".format(i)])
                for i in range(1, 50)])
        else:
            return zip(*enumerate([float(line.split()[-1])
                for line in f.readlines()]))

if __name__ == "__main__":
    ngram_path = sys.argv[1]
    unigram_path = sys.argv[2]
    model_path = sys.argv[3]
    datasets = {}
    datasets['trigram'] = load_data(ngram_path)
    datasets['unigram'] = load_data(unigram_path)
    datasets['model'] = load_data(model_path)
    plot(datasets)
