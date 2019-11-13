import sys
import os
import re
import itertools
from pathlib import Path
import xml.etree.cElementTree as etree

def doc_paths(data_dir, extension='.xml'):
    return [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(data_dir)
            for f in filenames if os.path.splitext(f)[1] == extension]

def path_to_paper_id(path):
    return '-'.join(os.path.basename(path).split('-')[:2])

def author_map(self, author):
    # maps alternate names for top 30 or so authors. Not perfect!
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

def get_abstract_table(paths):
    abstract_table = {}
    count = len(paths)
    for i, path in enumerate(paths):
        if i % 10 == 0:
            print(str(i) + " of " + str(count))
        tree = etree.iterparse(path, events=('start',))
        paper_id = path_to_paper_id(path)
        abs_section = False
        for event, elem in tree:
            if event == 'start':
                if elem.tag == 'sectionHeader' and elem.get('genericHeader') == 'abstract':
                    abs_section = True
                if abs_section == True and elem.tag == 'bodyText' and elem.text:
                    abstract_table[paper_id] = ' '.join(re.sub(r'-\n', '', elem.text).split())
                    break
    return abstract_table

if __name__ == "__main__":
    xml_dir  = sys.argv[1]
    abstract_table = get_abstract_table(xml_dir)
