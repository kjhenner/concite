#!/bin/python3
import sys
from collections import defaultdict
import pprint as pp

training_data_path = sys.argv[1]
count_output_path = sys.argv[2]

def to_ngrams(string, n):
    tokens = ['<s>'] + string.split() + ['</s>']
    return [tuple(tokens[i:i+n]) for i in range(0, len(tokens) - n + 1)]

def collect_ngrams(lines):
    ngrams = [defaultdict(int) for _ in range(0, 3)]
    for line in lines:
        for n in range(0, 3):
            for ngram in to_ngrams(line, n+1):
                ngrams[n][ngram] += 1
    return ngrams

with open(training_data_path) as f:
    ngram_counts = collect_ngrams(f.readlines())

with open(count_output_path, 'w') as f:
    lines = []
    for n in range(0, 3):
        items = sorted(list(ngram_counts[n].items()), key=lambda x : (-x[1],x[0]))
        lines += [str(count) + ' ' + ' '.join(ngram)
                for ngram, count in items]
    f.write("\n".join(lines))
