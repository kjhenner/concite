import sys
import os
import numpy as np
from sklearn.manifold import TSNE
from collections import defaultdict
import jsonlines

def get_emb_lookup(emb_path):
    emb_lookup = {}
    with open(emb_path) as f:
        for line in f.readlines():
            items = line.split()
            emb_lookup[items[0]] = np.array(list(map(float, items[1:])))
    return emb_lookup

if __name__ == "__main__":
    emb_path = sys.argv[1]
    data_path = sys.argv[2]
    n_components = int(sys.argv[3])
    output_path = sys.argv[4]

    examples = list(jsonlines.open(data_path))

    emb_lookup = get_emb_lookup(emb_path)
    examples = [example for example in examples if example['paper_id'] in emb_lookup.keys()]
    emb_matrix = [emb_lookup[example['paper_id']] for example in examples]
    tsne = TSNE(n_components=n_components, random_state=0).fit_transform(emb_matrix)

    with open(output_path, 'w') as f:
        for i, row in enumerate(tsne):
            f.write("{} {} {}\n".format(examples[i]['paper_id'], examples[i]['date'], ' '.join([str(x) for x in row])))
