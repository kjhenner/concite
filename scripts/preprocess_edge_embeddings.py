import sys
import networkx as nx
import jsonlines
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from concite.dataset_preprocessors import citation_graph

edge_path = sys.argv[1]
node_path = sys.argv[2]
output_path = sys.argv[3]

G = citation_graph.CitGraph()
G.load_edges(edge_path)
G.filter_by_degree()

emb = G.embed_edges(G.edges(), use_cache=True)

with jsonlines.open(node_path) as reader:
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(map(lambda x: dict({'n2v_vector':list(emb.to_vec(x['pmid']))}, **x), [ex for ex in reader if ex['pmid'] in G]))
