import jsonlines
import json
import sys

#paper_id, date, title, abstract, authors, last_author

def get_workshop_lookup(path):
    workshop_lookup = {}
    with open(path) as f:
        for line in f.readlines():
            items = line.split('\t')
            for paper_id in items[2:]:
                workshop_lookup[paper_id] = items[1]
    return workshop_lookup

def get_abstract_lookup(path):
    return {ex['paper_id']: ex['abstract'] for ex in jsonlines.open(path)}

def author_map(author):
    author_identities = {
        'C D Manning': 'Christopher D Manning',
        'C Manning': 'Christopher D Manning',
        'Christopher Manning': 'Christopher D Manning',
        'Jun’ichi Tsujii': "Jun'ichi Tsujii",
        'J Tsujii': "Jun'ichi Tsujii",
        'Jun´ıchi': "Jun'ichi Tsujii",
        'Junichi Tsujii': "Jun'ichi Tsujii",
        "E Hovy": "Eduard Hovy",
        "E H Hovy": "Eduard Hovy",
        "Eduard H Hovy": "Eduard Hovy",
        "D Klein": "Dan Klein",
        "D E Klein": "Dan Klein",
        "H Ney": "Hermann Ney",
        "Y Matsumoto": "Yuji Matsumoto",
        "D Roth": "Dan Roth",
        "D L Roth": "Dan Roth",
        "M Lapata": "Mirella Lapata",
        "K Knight": "Kevin Knight",
        "O Rambow": "Owen Rambow",
        "O C Rambow": "Owen Rambow",
        "Owen C Rambow": "Owen Rambow",
        "Noah Smith": "Noah A Smith",
        "N Smith": "Noah A Smith",
        "N A Smith": "Noah A Smith",
        "Y Liu": "Yang Liu",
        "H T Ng": "Hwee Tou Ng",
        "Hwee Ng": "Hwee Tou Ng",
        "Tou Hwee Ng": "Hwee Tou Ng",
        "P Koehn": "Philipp Koehn",
        "M Johnson": "Mark Johnson",
        "R Mihalcea": "Rada Mihalcea",
        "R F Mihalcea": "Rada Mihalcea",
        "S Kurohashi": "Sadao Kurohashi"
    }
    return author_identities.get(author, author)

def get_data_lookup(path, workshop_lookup, abstract_lookup, workshop_map):
    data_lookup = {}
    with open(path) as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            if len(items) > 3:
                workshop =  workshop_lookup.get(items[0][:6])
                combined_workshop = workshop_map.get(workshop, workshop)
                field_data = {
                    'paper_id': items[0],
                    'authors': items[-1].split(', '),
                    'last_author': author_map(items[-1].split(', ')[-1]),
                    'date': items[1],
                    'title': items[2],
                    'workshop': workshop,
                    'combined_workshop': combined_workshop,
                    'abstract': abstract_lookup.get(items[0])
                }
                data_lookup[items[0]] = field_data
    return data_lookup

def serialize_data(data_lookup, out_path):
    with open(out_path, 'w') as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(data_lookup.values())

def load_workshop_map(path):
    with open(path) as f:
        data = json.load(f)
    ret = {}
    for group, workshops in data.items():
        for workshop in workshops:
            ret[workshop] = group
    return ret


if __name__ == "__main__":
    abstract_path = sys.argv[1]
    workshop_path = sys.argv[2]
    workshop_map_path = sys.argv[3]
    canonical_data_path = sys.argv[4]
    out_path = sys.argv[5]

    abstract_lookup = get_abstract_lookup(abstract_path)
    workshop_map = load_workshop_map(workshop_map_path)
    workshop_lookup = get_workshop_lookup(workshop_path)

    data_lookup = get_data_lookup(canonical_data_path, workshop_lookup, abstract_lookup, workshop_map)
    serialize_data(data_lookup, out_path)
