import sys
import random
import json
import os
import jsonlines

edge_in = sys.argv[1]
node_in = sys.argv[2]
edge_out = sys.argv[3]

paper_ids = set()

with jsonlines.open(node_in) as reader:
    for obj in reader:
        paper_ids.add(obj['paper_id'])

with open(edge_in) as f:
    lines = f.readlines()

lines = [line for line in lines if all(map(lambda x: x in paper_ids, line.split()))]

with open(edge_out, 'w') as f:
    f.writelines(lines)
