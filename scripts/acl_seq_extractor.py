import sys
import os
import re
import itertools
import json
from pathlib import Path

def doc_paths(data_dir, extension='.json'):
    return [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(data_dir)
            for f in filenames if os.path.splitext(f)[1] == extension]

def load_json_paths(paths):
    sequences = []
    count = len(paths)
    for i, path in enumerate(paths):
        if i % 10 == 0:
            print(str(i) + " of " + str(count))
        load__path(path)

def load_id_seq(path):
    with open(path) as f:
        json_data = json.load(f)

    paper_ids = [[int(context['section']),
                  int(context['subsection']),
                  int(context['sentence']),
                  context['cited_paper_id']]
        for context in json_data['citation_contexts']]
    paper_ids.sort()
    return ' '.join([x[-1] for x in paper_ids])

if __name__ == "__main__":
    json_dir = sys.argv[1]
    out_path = sys.argv[2]

    paths = doc_paths(json_dir)

    sequences = []
    count = len(paths)
    for i, path in enumerate(paths):
        if i % 10 == 0:
            print(str(i) + " of " + str(count))
        sequences.append(load_id_seq(path))

    with open(out_path, 'w') as f:
        f.write('\n'.join(sequences))
