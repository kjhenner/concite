import sys
import os
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot(datasets):
    fig, ax = plt.subplots(figsize=(9,9))

    colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990', '#e6beff',
            '#9A6324', '#800000', '#aaffc3', '#808000', 'ffd8b']

    labels = []
    for i, (k, (X, Y)) in enumerate(datasets.items()):
        labels.append(k)
        ax.plot(X, Y, linewidth=0.8, color=colors[i])
    ax.legend(labels,loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=4, 
                        borderaxespad=0, frameon=False, fontsize='x-small')
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
    models = {
            'unigram': './output/unigram_metrics.out',
            'KN trigram': './output/trigram_metrics.out',
            'random init': './output/sequence/model.jsonl',
            'n2v all-p': './output/sequence/model_n2v_all_prop.jsonl',
            'n2v all-u': './output/sequence/model_n2v_all_uniform.jsonl',
            'n2v all-b': './output/sequence/model_n2v_bkg_penalty.jsonl',
            'n2v concat-f': './output/sequence/model_BERT_n2v_combined_fixed.jsonl',
            'n2v concat-p': './output/sequence/model_BERT_n2v_combined_prop.jsonl',
            'n2v concat-p': './output/sequence/model_BERT_n2v_combined_prop.jsonl',
            'SciBERT': './output/sequence/model_BERT.jsonl',
            'SciBERT n2v all-p': './output/sequence/model_BERT_n2v_all_prop.jsonl',
            'SciBERT n2v all-u': './output/sequence/model_BERT_n2v_all_uniform.jsonl',
            'SciBERT n2v all-b': './output/sequence/model_BERT_n2v_bkg_penalty.jsonl',
            'SciBERT n2v concat-f': './output/sequence/model_BERT_n2v_combined_fixed.jsonl',
            'SciBERT n2v concat-p': './output/sequence/model_BERT_n2v_combined_prop.jsonl',
            'SciBERT n2v concat-p': './output/sequence/model_BERT_n2v_combined_prop.jsonl'
        }
    datasets = {k: load_data(v) for k, v in models.items()}
    plot(datasets)
