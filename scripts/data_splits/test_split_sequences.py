import sys
import jsonlines
import random
import json
import os
from itertools import chain
from collections import defaultdict

def split_seq(seq, limit):
    if len(seq) > limit:
        return split_seq(seq[:int(len(seq)/2)], limit) + split_seq(seq[int(len(seq)/2):], limit)
    else:
        return [seq]

def validate_line(line, lookup):
    return len(line.split()) > 2 and all([lookup.get(pid) for pid in line.split()])

if __name__ == "__main__":

    input_file = sys.argv[1]
    train_prop = float(sys.argv[2])
    limit = int(sys.argv[3])

    counts = defaultdict(int)
    sequences = []
    with open(input_file) as f:
        for line in f.readlines():
            if len(line.strip().split()) > 3:
                sequences += split_seq(['<s>'] + line.strip().split(), 16)

    train_par = int(len(sequences) * train_prop)
    test_par = int((len(sequences) * (1-train_prop)) / 2)

    random.shuffle(sequences)

    train = sequences[:train_par]
    dev = sequences[train_par:-test_par]
    test = sequences[-test_par:]

    dirname, basename = os.path.split(input_file)

    print(len(train))
    print(len(dev))
    print(len(test))

    with open(os.path.join(dirname, 'train_'+basename), 'w') as f:
        f.write('\n'.join(map(lambda x: ' '.join(x), train)))

    with open(os.path.join(dirname, 'test_'+basename), 'w') as f:
        f.write('\n'.join(map(lambda x: ' '.join(x), test)))

    with open(os.path.join(dirname, 'dev_'+basename), 'w') as f:
        f.write('\n'.join(map(lambda x: ' '.join(x), dev)))
