import sys
import os
from multiprocessing import Pool, Manager
from pathlib import Path
sys.path.append(str(Path('.').absolute()))

from concite.dataset_preprocessors import pubmed_extractor

data_dir = sys.argv[1]
out_dir = sys.argv[2]

subdirs = [os.path.join(data_dir, name) for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))]

def extract_to(path):
    loader = pubmed_extractor.Loader(path)
    name_base = os.path.basename(path)

    documents, edges = loader.parse_paths()
    loader.write_edge_data(edges,
            os.path.join(out_dir, name_base + '_edges.jsonl'))
    loader.write_document_data(documents,
            os.path.join(out_dir, name_base + '_documents.jsonl'))

count = 0
with Pool(processes=50) as pool:
    for _ in pool.imap_unordered(extract_to, subdirs):
        print(count)
        count += 1
