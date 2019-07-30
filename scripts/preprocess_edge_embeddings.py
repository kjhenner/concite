import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from concite.dataset_preprocessors import citation_graph

data_path = sys.argv[1]
graph_out_path = sys.argv[2]

g = citation_graph.CitGraph()
g.load_data_from_dir(data_path)
g.save(graph_out_path)

#gv = g.degree_filtered_view()
#g.embed_edges(graph=gv,use_cache=False)
