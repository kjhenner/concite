import sys
import os
import re
import itertools
import jsonlines
import json
from pathlib import Path
from graph_tool.all import *
import xml.etree.cElementTree as etree

sys.path.append(str(Path('.').absolute()))
from concite.embedding.node2vec_wrapper import Node2VecEmb

class AclExtractor:

    def __init__(self, json_dir, xml_dir, workshop_map_path, workshop_min=3):
        self.workshop_min = workshop_min
        #self.title_map = self.load_title_map(paper_id_path)
        self.v_map = {}
        self.v_ids = {}
        self.load_workshop_map(workshop_map_path)
        self.g = Graph()
        self.vertex_fields = [
                ('paper_id', 'string'),
                ('abstract', 'string'),
                ('title', 'string'),
                ('venue', 'string'),
                ('graph_vector', 'vector<double>'),
                ('internal', 'boolean')]
        for pname, ptype in self.vertex_fields:
            self.g.vertex_properties[pname] = self.g.new_vertex_property(ptype)
        self.parse_xml_paths(self.doc_paths(xml_dir))
        self.load_json_paths(self.doc_paths(json_dir, extension='.json'))

    def load_title_map(self, path):
        with open(path) as f:
            return dict([(line[2], line[0])
                for line in map(lambda x: x.split('\t'), f.readlines())])

    def load_workshop_map(self, path):
        self.workshop_map = {}
        self.workshop_counts = {}
        with open(path) as f:
            workshops = map(lambda x: x.split('\t'), f.readlines())
        for workshop_fields in workshops:
            workshop_name = workshop_fields[1]
            self.workshop_counts[workshop_name] = len(workshop_fields[2:])
            for prefix in workshop_fields[2:]:
                if prefix:
                    self.workshop_map[prefix] = workshop_name

    def doc_paths(self, data_dir, extension='.xml'):
        return [os.path.join(dp, f)
                for dp, dn, filenames in os.walk(data_dir)
                for f in filenames if os.path.splitext(f)[1] == extension]

    def path_to_paper_id(self, path):
        return '-'.join(os.path.basename(path).split('-')[:2])

    def id2venue(self, paper_id):
       workshop = self.workshop_map.get(paper_id)
       if workshop and self.workshop_counts[workshop] >= self.workshop_min:
           return workshop
       else:
           return paper_id[0]

    def parse_xml_paths(self, paths):
        documents = []
        count = len(paths)
        for i, path in enumerate(paths):
            if i % 10 == 0:
                print(str(i) + " of " + str(count))
            documents.append(self.parse_xml(path))
        keys = [doc['paper_id'] for doc in documents]
        vertices = self.g.add_vertex(len(keys))
        vertex_indices = list(map(int, vertices))
        self.v_map.update(dict(zip(keys, vertex_indices)))
        self.v_ids.update(dict(zip(vertex_indices, keys)))
        for v, doc in zip(vertex_indices, documents):
            for field, _ in self.vertex_fields:
                if doc.get(field):
                    self.g.vp[field][v] = doc.get(field)

    def parse_xml(self, path):
        tree = etree.iterparse(path, events=('start',))
        paper_id =  self.path_to_paper_id(path)
        document_data = {
                'paper_id': paper_id,
                'title': None,
                'abstract': None,
                'venue': self.id2venue(paper_id),
                'internal': True
        }
        abs_section = False
        for event, elem in tree:
            if event == 'start':
                if elem.tag == 'title' and elem.text:
                    document_data['title'] = ' '.join(re.sub(r'-\n', '', elem.text).split())
                if elem.tag == 'sectionHeader' and elem.get('genericHeader') == 'abstract':
                    abs_section = True
                if abs_section == True and elem.tag == 'bodyText' and elem.text:
                    document_data['abstract'] = ' '.join(re.sub(r'-\n', '', elem.text).split())
                    break
        return document_data

    def ensure_vertex(self, v_id):
        v = self.v_map.get(v_id)
        if v is None:
            v = int(self.g.add_vertex())
            self.v_map[v_id] = v
            self.v_ids = v_id
            self.g.vp['paper_id'][v] = v_id
            self.g.vp['internal'][v] = False
        return v

    def load_json_paths(self, paths):
        count = len(paths)
        for i, path in enumerate(paths):
            if i % 10 == 0:
                print(str(i) + " of " + str(count))
            self.load_json_path(path)

    def load_json_path(self, path):
        with open(path) as f:
            json_data = json.load(f)
        citing_paper_id = json_data['paper_id']
        for citation_context in json_data['citation_contexts']:
            source_v = self.ensure_vertex(json_data['paper_id'])
            target_v = self.ensure_vertex(citation_context['cited_paper_id'])
            self.g.add_edge(source_v, target_v)

    def get_vertex_dict(self, v):
        v_dict = {}
        for pname, _ in self.vertex_fields:
            v_dict[pname] = self.g.vp[pname][v]
        v_dict['graph_vector'] = list(v_dict['graph_vector'])
        return v_dict

    def write_vertex_data(self, out_path):
        with jsonlines.open(out_path, 'w') as writer:
            writer.write_all([self.get_vertex_dict(v) for v in self.g.get_vertices()])

    def write_tsv_edge_list(self, out_path):
        with open(out_path, 'w') as f:
            for vertex in self.g.get_vertices():
                for neighbor in self.g.get_in_neighbors(vertex):
                    f.write("{}\t{}\n".format(self.g.vp['paper_id'][vertex], self.g.vp['paper_id'][neighbor]))
                for neighbor in self.g.get_out_neighbors(vertex):
                    f.write("{}\t{}\n".format(self.g.vp['paper_id'][vertex], self.g.vp['paper_id'][neighbor]))

    def embed_edges(self, l=40, d=128, p=0.5, q=0.5, name='acl', use_cache=False):
        emb = Node2VecEmb(self.g.get_edges()[:, :2], l, d, p, q, name=name, use_cache=use_cache)
        for i, vec in enumerate(emb.array):
            self.g.vp.graph_vector[int(self.g.vertex(emb.vector_idx_to_node[i]))] = vec

    def generate_walks(self, l=40, d=128, p=0.5, q=0.5, name='acl', use_cache=False):
        emb = Node2VecEmb(self.g.get_edges()[:, :2], l, d, p, q, True, name=name, use_cache=use_cache)
        print(emb.walks[0])
        return [[self.g.vp['paper_id'][v] for v in walk] for walk in emb.walks]

if __name__ == "__main__":
    json_dir = sys.argv[1]
    xml_dir  = sys.argv[2]
    workshop_map_path = sys.argv[3]
    vertex_out_path = sys.argv[4]
    edge_out_path = sys.argv[5]

    extractor = AclExtractor(json_dir, xml_dir, workshop_map_path)
    #extractor.write_tsv_edge_list(edge_out_path)
    extractor.embed_edges(ow=True)
    #extractor.write_vertex_data(vertex_out_path)
