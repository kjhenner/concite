#!/bin/python3
import math
import sys
from collections import defaultdict
import pprint as pp

ngram_count_path = sys.argv[1]
lm_output_path = sys.argv[2]

with open(ngram_count_path) as f:
    lines = f.readlines()

def format_output(output):
    return "{0} {1:.10f} {2:.10f} {3}".format(output[0], output[1], output[2], ' '.join(output[3]))

def build_lm(lines, out_path):
    type_totals = [0,0,0]
    token_totals = [0,0,0]
    ngram_counts = [{},{},{}]
    output = [[],[],[]]
    for line in lines:
        items = line.split()
        count = int(items[0])
        ngram = items[1:]
        n = len(ngram) - 1
        token_totals[n] += count
        type_totals[n] += 1
        ngram_counts[n][tuple(ngram)] = count
    for i in range(0, 3):
        for ngram, count in ngram_counts[i].items():
            n = len(ngram)
            if n == 1:
                p = float(count) / float(token_totals[0])
            else:
                p = float(count) / float(ngram_counts[i-1][tuple(ngram[:-1])])
            output[i].append((count, p, math.log(p, 10), ngram))
    output = [sorted(o, key=lambda x : (-x[0],x[3])) for o in output]
    lines = ["\\data\\"]
    for i in range(0, 3):
        lines.append("ngram {}: type={} token={}".format(i+1, type_totals[i], token_totals[i]))
    for i, ngram_list in enumerate(output):
        lines.append("")
        lines.append("\\{}-grams:".format(i+1))
        for line in ngram_list:
            lines.append(format_output(line))
    lines.append("")
    lines.append("\\end\\")
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))

build_lm(lines, lm_output_path)
