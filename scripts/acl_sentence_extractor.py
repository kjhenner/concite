import sys
import os
import itertools
import jsonlines
import json
from pathlib import Path

sys.path.append(str(Path('.').absolute()))

def load_json_paths(paths):
    count = len(paths)
    for i, path in enumerate(paths):
        if i % 10 == 0:
            print(str(i) + " of " + str(count))
        for instance in load_json_path(path):
            yield instance

def load_json_path(path):
    context_sentences = []
    with open(path) as f:
        json_data = json.load(f)
    citing_paper_id = json_data['paper_id']
    for citation_context in json_data['citation_contexts']:
        cited_paper_id = citation_context['cited_paper_id']
        section = citation_context["section"]
        subsection = citation_context["subsection"]
        sentence = citation_context["sentence"]
        sentence = json_data["sections"][section]["subsections"][subsection]["sentences"][sentence]["text"]
        yield {
            "text": sentence,
            "metadata": {
                "cited_paper": cited_paper_id,
                "citing_paper": citing_paper_id
            }
        }

if __name__ == "__main__":
    json_dir = sys.argv[1]
    out_path = sys.argv[2]
    paths = [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(json_dir)
            for f in filenames if os.path.splitext(f)[1] == '.json']
    with jsonlines.open(out_path, 'w') as writer:
        for instance in load_json_paths(paths):
            writer.write(instance)
