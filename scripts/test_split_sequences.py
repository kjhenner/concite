import sys
import jsonlines
import random
import json
import os
from itertools import chain
from collections import defaultdict

def split_seq(seq, limit, minimum=4):
    remainder = len(seq) % limit
    while 0 < remainder < minimum:
        limit -= 1
        remainder = len(seq) % limit
    if random.getrandbits(1):
        return zip(*[iter(seq)]*limit)
    else:
        return [seq[:remainder]] + list(zip(*[iter(seq[remainder:])]*limit))

def unpack_and_split(seqs, counts, limit, minimum=4):
    output = []
    for seq in seqs:
        for _ in range(counts[tuple(seq)]):
            output += split_seq(seq, limit, minimum)
    return [' '.join(seq) for seq in output]

def validate_line(line, lookup):
    return len(line.split()) > 2 and all([lookup.get(pid) and lookup[pid].get('abstract') for pid in line.split()])

if __name__ == "__main__":

    input_file = sys.argv[1]
    train_prop = float(sys.argv[2])
    limit = int(sys.argv[3])
#    lookup_path = sys.argv[4]
#
#    with jsonlines.open(lookup_path) as reader:
#        lookup = {
#            item['paper_id']: item for item in reader
#        }
#        lookup['<s>'] = {'abstract':'[unused0]'}
#        lookup['</s>'] = {'abstract':'[unused1]'}

    counts = defaultdict(int)
    with open(input_file) as f:
        sequences = [['<s>', *line.split(), '</s>']
                for line in f.readlines()]

    for seq in sequences:
        counts[tuple(seq)] += 1

    train_par = int(len(counts) * train_prop)
    test_par = int(len(counts)/2 * (1-train_prop))

    unique_sequences = list(counts.keys())
    random.shuffle(unique_sequences)

    train = unpack_and_split(unique_sequences[:train_par], counts, limit)
    dev = unpack_and_split(unique_sequences[train_par:-test_par], counts, limit)
    test = unpack_and_split(unique_sequences[-test_par:], counts, limit)

    dirname, basename = os.path.split(input_file)

    print(len(train))
    print(len(dev))
    print(len(test))

    with open(os.path.join(dirname, 'train_'+basename), 'w') as f:
        f.write('\n'.join(train))

    with open(os.path.join(dirname, 'test_'+basename), 'w') as f:
        f.write('\n'.join(test))

    with open(os.path.join(dirname, 'dev_'+basename), 'w') as f:
        f.write('\n'.join(dev))
