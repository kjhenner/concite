import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt

def plot(datasets):
    fig, ax = plt.subplots(figsize=(8,8))
    labels = []
    for i, (k, (X, Y)) in enumerate(datasets.items()):
        labels.append(k)
        ax.plot(X, Y, linewidth=0.8)
    ax.legend(labels,loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=3, 
                        borderaxespad=0, frameon=False)
    ax.set_xlabel("N")
    ax.set_ylabel("recall@N")
    plt.savefig('recall_plot.png')

def load_data(path):
    with open(path) as f:
        if path.split('.')[-1] == 'jsonl':
            data_dict = json.load(f)
            return zip(*[(i, data_dict["recall_at_{}".format(i)])
                for i in range(1, 50)])
        else:
            return zip(*enumerate([float(line.split()[-1])
                for line in f.readlines()]))

if __name__ == "__main__":
    ngram_path = './output/unigram_metrics.out'
    unigram_path = './output/trigram_metrics.out'
    scibert_path = './output/sequence/abstract.jsonl'
    scibert_n2v_path = './output/sequence/abstract_n2v_all_20_384_0.3_0.7.jsonl'
    scibert_n2v_intent_path = './output/sequence/abstract_n2v_combined_20_384_0.3_0.7_0.5.jsonl'
    n2v_path = './output/sequence/n2v_all_20_384_0.3_0.7.jsonl'
    n2v_intent_path = './output/sequence/n2v_combined_20_384_0.3_0.7_0.5.jsonl'
    datasets = {
        'KN trigram': load_data(ngram_path),
        'unigram': load_data(unigram_path),
        'SciBERT': load_data(scibert_path),
        'SciBERT + n2v': load_data(scibert_n2v_path),
        'SciBERT + n2v intent': load_data(scibert_n2v_intent_path),
        'n2v': load_data(n2v_path),
        'n2v intent': load_data(n2v_intent_path)
    }
    plot(datasets)
