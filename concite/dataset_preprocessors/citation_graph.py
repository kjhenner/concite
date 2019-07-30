import os
from graph_tool.all import *
import numpy as np
import random
from collections import defaultdict
from collections.abc import Iterable
import csv
import jsonlines

import sys
from pathlib import Path

sys.path.append(str(Path('.').absolute()))
from concite.embedding.node2vec_wrapper import Node2VecEmb

class CitGraph(Graph):

    def __init__(self,
            v_key = 'pmid',
            e1_key = 'citing_paper_id',
            e2_key = 'cited_paper_id'):
        super().__init__()
        self.fields = ['pmid', 'abstract', 'title', 'year', 'journal-id']
        for field in self.fields:
            self.vertex_properties[field] = self.new_vertex_property('string')
        self.v_key = v_key
        self.e1_key = e1_key
        self.e2_key = e2_key
        self.v_map = {}
        self.v_ids = {}

    def load_data_from_dir(self, data_path):
        self.load_vertices_from_dir(data_path)
        self.load_edges_from_dir(data_path)

    def load_vertices_from_dir(self, vertex_path):
        paths = [os.path.join(dp, f)
                for dp, dn, filenames in os.walk(vertex_path)
                for f in filenames if f.split('_')[-1] == 'documents.jsonl']
        path_count = len(paths)
        for i, path in enumerate(paths):
            print("loading {} of {} node files".format(i, path_count))
            self.load_vertices(path)

    def load_vertices(self, vertex_path):
        with jsonlines.open(vertex_path) as reader:
            documents = [doc for doc in reader
                    if doc.get(self.v_key) and doc.get('abstract') and doc.get('journal-id')]
        keys = [doc[self.v_key] for doc in documents]
        vertices = self.add_vertex(len(keys))
        # add_vertex only returns a generator if multiple vertices are added,
        # and otherwise returns the added vertex. In the latter case, we must
        # convert that single vertex to an iterator so it works with the
        # following code.
        if not isinstance(vertices, Iterable):
            vertex_indices = [int(vertices)]
        else:
            vertex_indices = list(map(int, vertices))
        self.v_map.update(dict(zip(keys, vertex_indices)))
        self.v_ids.update(dict(zip(vertex_indices, keys)))
        for v, doc in zip(vertex_indices, documents):
            for field in self.fields:
                self.vp[field][v] = doc[field]

    def load_edges_from_dir(self, edge_path):
        paths = [os.path.join(dp, f)
                for dp, dn, filenames in os.walk(edge_path)
                for f in filenames if f.split('_')[-1] == 'edges.jsonl']
        path_count = len(paths)
        for i, path in enumerate(paths):
            print("loading {} of {} edge files".format(i, path_count))
            self.load_edges(path)

    def ensure_vertex(self, v_id):
        v = self.v_map.get(v_id)
        if v is None:
            v = int(self.add_vertex())
            self.v_map[v_id] = v
            self.v_ids[v] = v_id
            self.vp[self.v_key][v] = v_id
        return v

    def load_edges(self, edge_path):
        edge_list = []
        with jsonlines.open(edge_path) as reader:
            for ex in reader:
                source_id = ex.get(self.e1_key)
                target_id = ex.get(self.e2_key)
                if source_id and target_id:
                    source_v = self.ensure_vertex(source_id)
                    target_v = self.ensure_vertex(target_id)
                    self.add_edge(source_v, target_v)

    def degree_filtered_view(self, min_degree=2):
        return GraphView(self, vfilt=lambda v : v.out_degree() + v.in_degree() > min_degree)

    def embed_edges(self, graph=None, l=40, d=128, p=0.5, q=0.5, name='emb', use_cache=True):
        if not graph:
            graph = self
        emb = Node2VecEmb(graph.get_edges()[:, :2], l, d, p, q, name=name, use_cache=use_cache)
        self.vp['graph_vector'] = self.new_vertex_property('vector<double>')
        for i, vec in enumerate(emb.array):
            self.vp.graph_vector[int(self.vertex(emb.vector_idx_to_node[i]))] = vec

