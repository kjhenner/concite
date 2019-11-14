import sys
import os
import itertools
import jsonlines
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.append(str(Path('.').absolute()))

def load_paths(paths):
    count = len(paths)
    sent_contexts = defaultdict(list)
    sec_contexts = defaultdict(list)
    for i, path in enumerate(paths):
        if i % 10 == 0:
            print(str(i) + " of " + str(count))
        with open(path) as f:
            json_data = json.load(f)
            citing_id = json_data['paper_id']
            for context in json_data['citation_contexts']:
                cited_id = context['cited_paper_id']
                sec = context["section"]
                subsec = context["subsection"]
                sent = context["sentence"]
                sent_contexts[(citing_id,sec,subsec,sent)].append(cited_id)
                sec_contexts[(citing_id,sec)].append(cited_id)
    sent_counts = defaultdict(int)
    sec_counts = defaultdict(int)
    #for context in sent_contexts.values():
    #    for k in map(lambda x: to_key(*x), combinations(context, 2)):
    #        sent_counts[k] +=1
    for context in sec_contexts.values():
        for k in map(lambda x: to_key(*x), combinations(context, 2)):
            sec_counts[k] +=1
    return (sent_counts, sec_counts)

def to_key(a, b):
    return tuple(sorted([a, b]))

if __name__ == "__main__":
    json_dir = sys.argv[1]
    sent_out_path = sys.argv[2]
    sec_out_path = sys.argv[3]
    paths = [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(json_dir)
            for f in filenames if os.path.splitext(f)[1] == '.json']
    sent_contexts, sec_contexts = load_paths(paths)
    #with open(sent_out_path, 'w') as f:
    #    for k, v in sent_contexts.items():
    #        f.write("{} {} {}\n".format(k[0], k[1], v))
    with open(sec_out_path, 'w') as f:
        for k, v in sec_contexts.items():
            f.write("{} {} {}\n".format(k[0], k[1], v))
