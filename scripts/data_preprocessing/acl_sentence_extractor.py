import sys
import os
import itertools
import jsonlines
import json
from pathlib import Path

sys.path.append(str(Path('.').absolute()))

def load_json_paths(paths, intent_lookup):
    count = len(paths)
    for i, path in enumerate(paths):
        if i % 10 == 0:
            print(str(i) + " of " + str(count))
        for instance in load_json_path(path, intent_lookup):
            yield instance

def load_json_path(path, intent_lookup):
    context_sentences = []
    with open(path) as f:
        json_data = json.load(f)
    citing_paper_id = json_data['paper_id']
    for citation_context in json_data['citation_contexts']:
        citing_string = citation_context['citing_string']
        cited_paper_id = citation_context['cited_paper_id']
        is_self_cite = citation_context['is_self_cite']
        section = citation_context["section"]
        subsection = citation_context["subsection"]
        sentence = citation_context["sentence"]
        sentence_text = json_data["sections"][section]["subsections"][subsection]["sentences"][sentence]["text"]
        intent_label = intent_lookup[(citing_paper_id, cited_paper_id)]['label']
        yield {
            "text": sentence_text,
            "metadata": {
                "citing_string": citing_string,
                "section_number": section,
                "is_self_cite": is_self_cite,
                "intent_label": intent_label,
                "cited_paper": cited_paper_id,
                "citing_paper": citing_paper_id
            }
        }

if __name__ == "__main__":
    json_dir = sys.argv[1]
    intent_path = sys.argv[2]
    out_path = sys.argv[3]
    with open(intent_path) as f:
        intent_lookup = {
            (ex['citing_paper'], ex['cited_paper']): ex
            for ex in jsonlines.Reader(f)
        }
    paths = [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(json_dir)
            for f in filenames if os.path.splitext(f)[1] == '.json']
    with jsonlines.open(out_path, 'w') as writer:
        for instance in load_json_paths(paths, intent_lookup):
            writer.write(instance)
