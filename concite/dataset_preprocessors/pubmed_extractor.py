import sys
import os
import pprint
import json
import itertools
import csv
import jsonlines
from collections import defaultdict
import re
import xml.etree.cElementTree as etree

class Loader():

    def __init__(self, data_dir):
        self.doc_paths = self.get_doc_paths(data_dir)

    def get_doc_paths(self, data_dir):
        return [os.path.join(dp, f)
                for dp, dn, filenames in os.walk(data_dir)
                for f in filenames if os.path.splitext(f)[1] == '.nxml']

    def parse_paths(self):
        documents = []
        edges = []
        count = len(self.doc_paths)
        for i, path in enumerate(self.doc_paths):
            if i % 10 == 0:
                print(str(i) + " of " + str(count))
            try:
                document, cit_edges = self.parse_cit_contexts(path)
                documents.append(document)
                edges += cit_edges
            except Exception as e:
                print(path)
                print(e)
        return documents, edges

    def get_context_window(self, text, offset, window_size):
        start = max([offset - window_size, 0])
        end = min([offset + window_size, len(text)])
        return text[start:end]

    def rid2int(self, rid):
        m = re.match(r'(C|CIT|c|r|cr|CR|cit|ref|b|bibr|R|B)(\d+).*', rid)
        if m is not None and m.group(2) is not None:
            return int(m.group(2))
        else:
            m = re.match(r'.*(C|CIT|c|r|cr|CR|cit|ref|b|bibr|R|B)[\-_]?(\d+)$', rid)
            if m is not None and m.group(2) is not None:
                return int(m.group(2))

    def article_to_tsv_row(self, document, key=None):
        return '\t'.join(map(str, [
            document.get('pmid'),
            document.get('title'),
            document.get('year'),
            document.get('keywords')]))

    def edge_to_tsv_row(self, edge, key=None):
        return '\t'.join([
            edge['citing_paper_id'],
            edge['cited_paper_id']])

    def parse_cit_contexts(self, path):
        tree = etree.iterparse(path, events=('start', 'end'))
        document = {}
        in_body = False
        following_xref = False
        prev_xref_rid = None
        offset = 0
        text = ''
        section = ''
        rid = ''
        ref_dict = {}
        offset_dict = defaultdict(list)
        document['keywords'] = ''
        for event, elem in tree:
            if event == 'start':
                if elem.tag == 'body':
                    in_body = True
                if elem.text is not None and in_body:
                    offset += len(elem.text)
                    text += elem.text
                if elem.tag == 'journal-id' and elem.get('journal-id-type') == 'nlm-ta':
                    document['journal-id'] = elem.text
                if elem.tag == 'article-id' and elem.get('pub-id-type') == 'pmc':
                    document['pmc'] = elem.text
                if elem.tag == 'article-id' and elem.get('pub-id-type') == 'pmid':
                    document['pmid'] = elem.text
                if elem.tag == 'article-title':
                    document['title'] = ' '.join(elem.itertext())
                if elem.tag == 'abstract':
                    document['abstract'] = ' '.join(elem.itertext())
                if elem.tag == 'year':
                    document['year'] = elem.text
                if elem.tag == 'kwd':
                    document['keywords'] += ' ' + str(elem.text)
                if elem.tag == 'kwd':
                    document['keywords'] += ' ' + str(elem.text)
                if elem.tag == 'ref':
                    rid = self.rid2int(elem.get('id'))
                    ref_dict[rid] = {}
                if rid != '':
                    if elem.tag == 'article-title':
                        ref_dict[rid]['title'] = ''.join(elem.itertext())
                    if rid and elem.tag == 'year':
                        ref_dict[rid]['year'] = elem.text
                    if elem.tag == 'pub-id' and elem.get('pub-id-type') == 'pmid':
                        ref_dict[rid]['pmid'] = elem.text
            if event == 'end' and in_body == True:
                if elem.tag == 'body':
                    in_body = False
                if elem.tag == 'title':
                    title = elem.itertext()
                if elem.tag == 'xref' and elem.get('ref-type') == 'bibr':
                    for id_part in elem.get('rid').split(' '):
                        offset_dict[self.rid2int(id_part)].append(offset)
                        if prev_xref_rid:
                            for id in range(prev_xref_rid + 1, self.rid2int(id_part)):
                                offset_dict[id].append(offset)
                if elem.tail is not None:
                    offset += len(elem.tail)
                    text += elem.tail
                if elem.tag == 'xref' and elem.get('ref-type') == 'bibr' and elem.tail == '-':
                    prev_xref_rid = self.rid2int(elem.get('rid'))
                else:
                    prev_xref_rid = None
        return (document, self.get_edges(offset_dict, ref_dict, document, text))

    def get_edges(self, offset_dict, ref_dict, document, text):
        edges = []
        for rid, offsets in offset_dict.items():
            # Some papers are simply missing bibliography ref entries for some
            # xrefs
            if ref_dict.get(rid):
                for offset in offsets:
                    edges.append({
                        'citation_id': document['pmid'] + '_' + str(rid),
                        'cite_context': self.get_context_window(text, offset, 300),
                        'citing_paper_id': document['pmid'],
                        'citing_paper_title': document['title'],
                        'citing_paper_year': document['year'],
                        'cited_paper_id': ref_dict[rid].get('pmid'),
                        'cited_paper_title': ref_dict[rid].get('title'),
                        'cited_paper_year': ref_dict[rid].get('year')
                        })
        return edges

    def write_edge_data(self, edge_data, out_path):
        with jsonlines.open(out_path, 'w') as writer:
            writer.write_all(edge_data)

    def write_document_data(self, document_data, out_path):
        with jsonlines.open(out_path, 'w') as writer:
            writer.write_all(document_data)

if __name__ == "__main__":

    data_dir = sys.argv[1]
    documents_out = sys.argv[2]
    edges_out = sys.argv[3]

    loader = Loader(data_dir)
    documents, edges = loader.parse_paths()
    documents = [document for document in documents if document.get('pmid')]
    ids = set(document['pmid'] for document in documents)
    edges = [edge for edge in edges if edge.get('citing_paper_id') in ids and edge.get('cited_paper_id') in ids]
    loader.write_edge_data(edges, edges_out)
    loader.write_document_data(documents, documents_out)
