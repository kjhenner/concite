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
from spacy.lang.en import English

class Loader():

    def __init__(self, data_dir):
        self.doc_paths = self.get_doc_paths(data_dir)
        self.nlp = English()
        sentencizer = self.nlp.create_pipe("sentencizer")
        self.nlp.add_pipe(sentencizer)

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
                pass
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
            document.get('abstract'),
            document.get('year'),
            document.get('keywords')]))

    def edge_to_tsv_row(self, edge, key=None):
        return '\t'.join([
            edge['citing_paper_id'],
            edge['cited_paper_id'],
            edge['cite_sentence']])

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
                    offset += len(elem.tail) + 1
                    text += elem.tail + ' '
                if elem.tag == 'xref' and elem.get('ref-type') == 'bibr' and elem.tail == '-':
                    prev_xref_rid = self.rid2int(elem.get('rid'))
                else:
                    prev_xref_rid = None
        return (document, self.get_edges(offset_dict, ref_dict, document, text))

    def get_edges(self, offset_dict, ref_dict, document, text, char_window=600):
        edges = []
        for rid, offsets in offset_dict.items():
            # Some papers are simply missing bibliography ref entries for some
            # xrefs
            if ref_dict.get(rid):
                for offset in offsets:
                    edges.append({
                        'citation_id': document['pmid'] + '_' + str(rid),
                        'cite_sentence': self.mid_sentence(self.get_context_window(text, offset, char_window)),
                        'citing_paper_id': document['pmid'],
                        'cited_paper_id': ref_dict[rid].get('pmid'),
                        })
        return edges

    def write_edge_data(self, edge_data, out_path):
        with jsonlines.open(out_path, 'w') as writer:
            writer.write_all(edge_data)

    def write_document_data(self, document_data, out_path):
        with jsonlines.open(out_path, 'w') as writer:
            writer.write_all(document_data)

    def mid_sentence(self, string):
        mid = len(string)/2
        pos = 0
        for sent in self.nlp(re.sub(r'\[\.', ']. ', string)).sents:
            if pos + len(str(sent)) > mid:
                return str(sent)
            else:
                pos += len(str(sent))

if __name__ == "__main__":

    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    loader = Loader(data_dir)
    articles, edges = loader.parse_paths()
    print(len(articles))
    articles = [article for article in articles if article.get('pmid')]
    ids = set(article['pmid'] for article in articles)
    edges = [edge for edge in edges if edge.get('citing_paper_id') in ids and edge.get('cited_paper_id') in ids]

    loader.write_edge_data(edges, os.path.join(out_dir, 'pubmed_edges.jsonl'))
    loader.write_document_data(articles, os.path.join(out_dir, 'pubmed_articles.jsonl'))

#    with open(os.path.join(out_dir, 'pubmed_articles.tsv'), 'w') as f:
#        f.write('\n'.join([loader.article_to_tsv_row(article) for article in articles]))
#
#    with open(os.path.join(out_dir, 'pubmed_edges.tsv'), 'w') as f:
#        f.write('\n'.join([loader.edge_to_tsv_row(edge) for edge in edges]))
