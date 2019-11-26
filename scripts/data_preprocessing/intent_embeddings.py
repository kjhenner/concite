import sys
import jsonlines
from collections import defaultdict
import os
from pathlib import Path

sys.path.append(str(Path('.').absolute()))
from concite.embedding.node2vec_wrapper import Node2VecEmb

def load_edges(edge_path, weights):
    with jsonlines.open(edge_path) as reader:
        return [(ex['metadata']['citing_paper'],
                 ex['metadata']['cited_paper'],
                 weights[ex['metadata']['intent_label']]) for ex in reader]

def get_prop_wts(edge_path):
    with jsonlines.open(edge_path) as reader:
        examples = list(reader)
    intent_counts = defaultdict(int)
    for ex in examples:
        intent_counts[ex['metadata']["intent_label"]] += 1
    return {k: 1-v/float(len(examples))
            for k, v in intent_counts.items()}

def combine_embeddings(out_dir, suffix):
    combined_edges = defaultdict(list)
    for intent in ['method', 'background', 'result']:
        with open(os.path.join(out_dir, "{}_{}".format(intent, suffix))) as f:
            for line in f.readlines():
                items = line.strip().split()
                combined_edges[items[0]] += items[1:]
    with open(os.path.join(out_dir, "combined_{}".format(suffix)), 'w') as f:
        for k, v in combined_edges.items():
            f.write("{} {}\n".format(k, ' '.join(v)))

if __name__ == "__main__":
    edge_path = sys.argv[1]
    out_dir = sys.argv[2]

    d, p, q, l = 128, 0.3, 0.7, 20

    # Combined proportional
    prop_wts = get_prop_wts(edge_path)
    suffix = "{}_{}_{}_{}_{}.emb".format(d, p, q, l, 'prop')
    for intent in ['method', 'background', 'result']:
        wts = prop_wts.copy()
        wts[intent] = 1.0
        name = "{}_{}".format(intent, suffix)
        edges = load_edges(edge_path, weights=wts)
        emb = Node2VecEmb(edges, l, d, p, q, w=True)
        emb.write_embeddings(os.path.join(out_dir, name))
    combine_embeddings(out_dir, suffix)

    # Combined fixed-weight (0.5)
    wts = {'method': 0.5, 'background': 0.5, 'result': 0.5}
    suffix = "{}_{}_{}_{}_{}.emb".format(d, p, q, l, 'fixed')
    for intent in ['method', 'background', 'result']:
        wts = wts.copy()
        wts[intent] = 1.0
        name = "{}_{}".format(intent, suffix)
        edges = load_edges(edge_path, weights=wts)
        emb = Node2VecEmb(edges, l, d, p, q, w=True)
        emb.write_embeddings(os.path.join(out_dir, name))
    combine_embeddings(out_dir, suffix)

    # Full porportional
    name = "{}_{}_{}_{}_{}.emb".format('prop', 384, p, q, l)
    edges = load_edges(edge_path, weights=prop_wts)
    emb = Node2VecEmb(edges, l, d, p, q, w=True)
    emb.write_embeddings(os.path.join(out_dir, name))

    # Full drop background
    name = "{}_{}_{}_{}_{}.emb".format('bkg', 384, p, q, l)
    wts = {
        'method':1.0,
        'background':0.2,
        'result':1.0
    }
    edges = load_edges(edge_path, weights=wts)
    emb = Node2VecEmb(edges, l, d, p, q, w=True)
    emb.write_embeddings(os.path.join(out_dir, name))

    # Full uniform
    name = "{}_{}_{}_{}_{}.emb".format('uniform', 384, p, q, l)
    wts = {
        'method':1.0,
        'background':1.0,
        'result':1.0
    }
    edges = load_edges(edge_path, weights=wts)
    emb = Node2VecEmb(edges, l, d, p, q, w=True)
    emb.write_embeddings(os.path.join(out_dir, name))
