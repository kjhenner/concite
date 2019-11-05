import sys
import random
import os
import jsonlines
from collections import defaultdict

def get_counts(examples, key):
    counts = defaultdict(int)
    for ex in examples:
        value = ex.get(key)
        if value:
            counts[value] += 1
    counts = list(counts.items())
    counts.sort(key=lambda x: -x[1])
    return counts

input_file = sys.argv[1]
train_prop = float(sys.argv[2])
filter_key = sys.argv[3]
top_n = int(sys.argv[4])

examples = [ex for ex in list(jsonlines.open(input_file)) if ex['abstract']]

top_counts = get_counts(examples, filter_key)[:top_n]


top_values = [w[0] for w in top_counts]
examples = [ex for ex in examples if ex[filter_key] in top_values]

random.shuffle(examples) 
example_count = len(examples)

train_par = int(example_count * train_prop)
test_par = int(example_count/2 * (1-train_prop))

split_data = {
    'train': examples[:train_par],
    'dev': examples[train_par:-test_par],
    'test': examples[-test_par:]
}

dirname, basename = os.path.split(input_file)

print("Overall counts")
print(len(examples))
for count in top_counts:
    print(count)

for k, v in split_data.items():
    print("{} counts".format(k))
    print(len(v))
    counts = get_counts(v, filter_key)
    for count in counts:
        print(count)
    path = os.path.join(dirname, '{}_{}_{}_{}'.format(k, top_n, filter_key, basename))
    with open(path, 'w') as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(v)
