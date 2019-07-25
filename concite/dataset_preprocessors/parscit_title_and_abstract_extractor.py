import sys
import os
import re
import itertools
import jsonlines
import xml.etree.cElementTree as etree

data_dir = sys.argv[1]
out_path = sys.argv[2]

class ParscitTitleAndAbstractExtractor:

    def __init__(self, data_dir):
        self.paths = self.doc_paths(data_dir)[:10]
        self.documents = self.parse_paths()

    def doc_paths(self, data_dir):
        return [os.path.join(dp, f)
                for dp, dn, filenames in os.walk(data_dir)
                for f in filenames if os.path.splitext(f)[1] == '.xml']

    def path_to_paper_id(self, path):
        return '-'.join(os.path.basename(path).split('-')[:2])

    def parse_paths(self):
        documents = []
        count = len(self.paths)
        for i, path in enumerate(self.paths):
            if i % 10 == 0:
                print(str(i) + " of " + str(count))
            documents.append(self.parse(path))
        return documents

    def parse(self, path):
        tree = etree.iterparse(path, events=('start',))
        document_data = {
                'paper_id': self.path_to_paper_id(path),
                'title': None,
                'abstract': None
        }
        abs_section = False
        for event, elem in tree:
            if event == 'start':
                if elem.tag == 'title':
                    document_data['title'] = ' '.join(re.sub(r'-\n', '', elem.text).split())
                if elem.tag == 'sectionHeader' and elem.get('genericHeader') == 'abstract':
                    abs_section = True
                if abs_section == True and elem.tag == 'bodyText':
                    document_data['abstract'] = ' '.join(re.sub(r'-\n', '', elem.text).split())
                    break
        return document_data

    def write_document_data(self, out_path):
        with jsonlines.open(out_path, 'w') as writer:
            writer.write_all(self.documents)

if __name__ == "__main__":
    extractor = ParscitTitleAndAbstractExtractor(data_dir)
    extractor.write_document_data(out_path)
