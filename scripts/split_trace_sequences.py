from itertools import chain
import random
import jsonlines
import csv
import sys
import os

def split_seq(seq, limit, minimum=4):
    remainder = len(seq) % limit
    while remainder < minimum:
        limit -= 1
        remainder = len(seq) % limit
    if random.getrandbits(1):
        return zip(*[iter(seq)]*limit)
    else:
        return seq[:remainder] + zip(*[iter(seq[remainder:])]*limit)

if __name__ == "__main__":
    sequence_path = sys.argv[1]
    lookup_path = sys.argv[2]
    sequence_limit = int(sys.argv[3])
    output_path = sys.argv[5]

    with jsonlines.open(lookup_path) as reader:
        data_lookup = {
            item['paper_id']: item for item in reader
        }
        data_lookup['<s>'] = {'abstract':'<s>'}

    out_lines = []
    with open(sequence_path) as f:
        for ex in f.readlines():
            trace_seq = ['<s>', *ex.split()]
            else:
                split_seqs = [trace_seq]
            for split_seq in split_seqs:
                if len(split_seq) > 1 and all([data_lookup.get(paper_id)) for paper_id in split_seq]):
                    out_lines.append(' '.join(split_seq))

    with open(output_path, 'w') as f:
        f.write('\n'.join(out_lines))
