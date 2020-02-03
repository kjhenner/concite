from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from itertools import chain
from collections import defaultdict
from random import shuffle
import numpy as np
import math
import jsonlines
import csv
import sys
import os

class recall_at_n:

    def __init__(self, n):
        self.n = n
        self.count = 0
        self.tp_count = 0

    def __call__(self, targets, predictions):
        for i, target in enumerate(targets):
            self.count += 1
            if target in predictions[i][:self.n]:
                self.tp_count += 1

    def get_metric(self):
        return float(self.tp_count) / self.count

def train_ngram_lm(bin_path, train_path, model_path, order='3', model='ukndiscount'):
    args = [ os.path.join(bin_path, 'ngram-count'),
             '-text',
             train_path,
             '-' + model,
             '-order',
             order,
             '-no-sos',
             '-no-eos',
             '-lm',
             model_path
           ]
    p = Popen(args, stdout=PIPE)
    print(p.communicate()[0].decode('utf-8'))

def test_ngram_lm(bin_path, test_path, model_path, order='3'):
    args = [ os.path.join(bin_path, 'ngram'),
             '-ppl',
             test_path,
             '-order',
             order,
             '-no-sos',
             '-no-eos',
             '-lm',
             model_path
           ]
    p = Popen(args, stdout=PIPE)
    print(p.communicate()[0].decode('utf-8'))

def load_lm(path):
    with open(path) as f:
        lines = f.readlines()
    lm = {}
    for line in lines[6:]:
        if len(line) > 1 and line[0] != '\\':
            items = line.split()
            try:
                lm[tuple(items[1:-1])] = (float(items[0]), float(items[-1]))
            except ValueError:
                lm[tuple(items[1:])] = (float(items[0]),)
    return lm

def ngram_prob(lm, ngram):
    if not len(ngram):
        return -99
    lprobs = lm.get(ngram)
    if lprobs:
        return sum(lprobs)
    return ngram_prob(lm, tuple(ngram[1:]))

def predict_sequences(test_path, lm, vocab, metrics, unigram_only=False):
    k = max([metric.n for metric in metrics])
    with open(test_path) as f:
        sequences = [line.split() for line in f.readlines()]
    total = len(sequences)
    out = []
    top_n_counts = defaultdict(int)
    shuffle(sequences)
    for i, sequence in enumerate(sequences[:100]):
        print("{} of {}".format(i, total))
        targets = sequence[1:]
        preds = predict_sequence(sequence, lm, vocab, k)
        for metric in metrics:
            metric(targets, preds)

def predict_sequence(sequence, lm, vocab, k, unigram_only=False):
    return [topk(tuple(sequence[i-1:i]), lm, vocab, k)
            for i in range(1, len(sequence))]

def topk(prefix, lm, vocab, k, unigram_only=False):
    if unigram_only:
        probs = np.array([ngram_prob(lm, (token,))
                for token in vocab])
    else:
        probs = np.array([ngram_prob(lm, tuple([*prefix, token]))
                for token in vocab])
    ind = np.argpartition(probs, -k)[-k:]
    ind = ind[np.argsort(probs[ind])]
    return [vocab[i] for i in ind]
        
def get_vocab(train_path):
    vocab = set()
    with open(train_path) as f:
        for line in f.readlines():
            for item in line.split():
                vocab.add(item)
    return list(vocab)

if __name__ == "__main__":
    bin_path = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    metric_out_path = sys.argv[4]
    unigram_only = sys.argv[5]
    if unigram_only == 'true':
        unigram_only = True
        order = '1'
    else:
        unigram_only = False
        order = '3'

    with NamedTemporaryFile(mode='w') as f:
        model_path = f.name
        train_ngram_lm(bin_path, train_path, model_path, order=order)
        test_ngram_lm(bin_path, test_path, model_path, order=order)
#        vocab = get_vocab(train_path)
#        lm = load_lm(model_path)
#        metrics = [recall_at_n(n) for n in range(1, 50)]
#        predict_sequences(test_path, lm, vocab, metrics, unigram_only)
#        with open(metric_out_path, 'w') as f:
#            for metric in metrics:
#                f.write("{}: {}\n".format(metric.n, metric.get_metric()))
