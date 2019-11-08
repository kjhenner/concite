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

def train_dev_test_split(examples, train_prop, input_file, filter_key):

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
        path = os.path.join(dirname, '{}_{}_{}'.format(k, filter_key, basename))
        print(path)
        write_examples(v, path)
    
if __name__ == "__main__":
    input_file = sys.argv[1]
    paper_data_path = sys.argv[2]
    filter_field = sys.argv[3]
    train_prop = float(sys.argv[4])

    with open(paper_data_path) as f:
        paper_lookup = {ex['paper_id']: ex for ex in jsonlines.Reader(f)}

    with open(input_file) as f:
        examples = list(jsonlines.Reader(f))

    examples = [
        ex for ex in examples if paper_lookup.get(ex['metadata']['citing_paper'],{}).get(filter_field) and paper_lookup.get(ex['metadata']['cited_paper'],{}).get(filter_field)
    ]

    random.shuffle(examples) 
    example_count = len(examples)

    train_dev_test_split(examples, train_prop, input_file, filter_field)
