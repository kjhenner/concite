import sys
import random
import os
import jsonlines
from collections import defaultdict
from itertools import chain

def write_examples(examples, path):
    with open(path, 'w') as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(examples)

def write_k_fold_examples(sets, folds, top_n, filter_key, input_file):
    dirname, basename = os.path.split(input_file)
    write_dir = os.path.join(dirname, '{}_{}_{}-fold'.format(top_n, filter_key, folds))
    try:
        os.mkdir(write_dir)
    except FileExistsError:
        pass
    path = os.path.join(write_dir, 'test_{}'.format(basename))
    write_examples(sets['test'], path)
    for i, train_set in enumerate(sets['train']):
        path = os.path.join(write_dir, 'train_{}_{}'.format(i, basename))
        write_examples(train_set, path)
    for i, validation_set in enumerate(sets['validation']):
        path = os.path.join(write_dir, 'validation_{}_{}'.format(i, basename))
        write_examples(validation_set, path)

def k_fold_split(examples, k):

    n, m = divmod(len(examples), k)
    splits = [examples[i * n + min(i, m):(i + 1) * n + min(i + 1, m)] for i in range(k)]
    sets = {
        'test': splits[0],
        'train': [],
        'validation': [],
    }
    for i in range(1, k):
        sets['train'].append(examples[i])
        validation = []
        for j in range(1, k):
            if j != i:
                validation += splits[j]
        sets['validation'].append(validation)
    return sets

def train_dev_test_split(examples, train_prop):

    dirname, basename = os.path.split(input_file)

    train_par = int(example_count * train_prop)
    test_par = int(example_count/2 * (1-train_prop))

    split_data = {
        'train': examples[:train_par],
        'dev': examples[train_par:-test_par],
        'test': examples[-test_par:]
    }


    for k, v in split_data.items():
        print("{} counts".format(k))
        print(len(v))
        counts = get_counts(v, filter_key)
        for count in counts:
            print(count)
        path = os.path.join(dirname, '{}_{}_{}_{}'.format(k, top_n, filter_key, basename))
        write_examples(v, path)
    
def get_counts(examples, key):
    counts = defaultdict(int)
    for ex in examples:
        value = ex.get(key)
        if value:
            counts[value] += 1
    counts = list(counts.items())
    counts.sort(key=lambda x: -x[1])
    return counts

if __name__ == "__main__":
    input_file = sys.argv[1]
    split_type = sys.argv[2]
    filter_key = sys.argv[3]
    top_n = int(sys.argv[4])

    examples = [ex for ex in list(jsonlines.open(input_file)) if ex['abstract']]

    top_counts = get_counts(examples, filter_key)[:top_n]

    top_values = [w[0] for w in top_counts]
    examples = [ex for ex in examples if ex[filter_key] in top_values]

    random.shuffle(examples) 
    example_count = len(examples)


    print("Overall counts")
    print(len(examples))
    for count in top_counts:
        print(count)

    if split_type.split('-')[-1] == 'fold':
        k = int(split_type.split('-')[0])
        k_fold_examples = k_fold_split(examples, k)
        write_k_fold_examples(k_fold_examples, k, top_n, filter_key, input_file)
    else:
        train_prop = int(split_type)
        train_dev_test_split(examples, train_prop, input_file)
