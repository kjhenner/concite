import sys
import networkx as nx
import jsonlines
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from concite.dataset_preprocessors import pubmed_extractor

data_dir = sys.argv[1]
documents_out = sys.argv[2]
edges_out = sys.argv[3]

loader = pubmed_extractor.Loader(data_dir)
documents, edges = loader.parse_paths()
documents = [document for document in documents if document.get('pmid') and document.get('abstract')]
# Uncommet to limit edges to internal documents
#ids = set(document['pmid'] for document in documents)
#edges = [edge for edge in edges if edge.get('citing_paper_id') in ids and edge.get('cited_paper_id') in ids]
loader.write_edge_data(edges, edges_out)
loader.write_document_data(documents, documents_out)
