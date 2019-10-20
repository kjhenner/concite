import sys
import jsonlines
from collections import defaultdict
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
    smoothed_wt=0.1
    l = 40
    p = 0.5
    q = 0.5
    for intent, edges in edges_from_intent_jsonl(edge_path, smoothed_wt).items():
        d = 128
        if intent == 'all':
            d = 3*d
        emb = Node2VecEmb(edges, l, d, p, q, w=True, use_cache=False, name=intent)
        emb.write_embeddings("{}_{}_{}_{}_{}_{}.emb".format(intent, l, d, p, q, smoothed_wt))
