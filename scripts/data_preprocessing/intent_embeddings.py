import sys
import jsonlines
from collections import defaultdict
import os
from pathlib import Path

sys.path.append(str(Path('.').absolute()))
from concite.embedding.node2vec_wrapper import Node2VecEmb

def edges_from_intent_jsonl(path, smoothed_wt=0.1):
    edges = defaultdict(list)
    with jsonlines.open(edge_path) as reader:
        for ex in reader:
            for intent in ['method', 'background', 'result']:
                if ex["label"] == intent:
                    wt = 1.0
                else:
                    wt = smoothed_wt
                edges[intent].append((ex['citing_paper'], ex['cited_paper'], wt))
            edges['all'].append((ex['citing_paper'], ex['cited_paper'], 1.0))
    return edges

if __name__ == "__main__":
    edge_path = sys.argv[1]
    out_dir = sys.argv[2]
    p = 0.3
    q = 0.7
    for l in [15,20,25]:
        for smoothed_wt in [0.5, 0.4, 0.3]:
            for intent, edges in edges_from_intent_jsonl(edge_path, smoothed_wt).items():
                d = 128
                if intent == 'all':
                    d = 3*d
                emb = Node2VecEmb(edges, l, d, p, q, w=True, use_cache=False, name=intent)
                if intent == 'all':
                    emb.write_embeddings(os.path.join(out_dir, "{}_{}_{}_{}_{}.emb".format(intent, l, d, p, q)))
                else:
                    emb.write_embeddings(os.path.join(out_dir, "{}_{}_{}_{}_{}_{}.emb".format(intent, l, d, p, q, smoothed_wt)))
            combined_edges = defaultdict(list)
            for intent in ['method', 'background', 'result']:
                d = 128
                with open(os.path.join(out_dir, "{}_{}_{}_{}_{}_{}.emb".format(intent, l, d, p, q, smoothed_wt))) as f:
                    for line in f.readlines():
                        items = line.strip().split()
                        combined_edges[items[0]] += items[1:]
            with open(os.path.join(out_dir, "combined_{}_{}_{}_{}_{}.emb".format(l, 3*d, p, q, smoothed_wt)), 'w') as f:
                for k, v in combined_edges.items():
                    f.write("{} {}\n".format(k, ' '.join(v)))
