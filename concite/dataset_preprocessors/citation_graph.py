import os
import networkx as nx
import numpy as np
import random
from collections import defaultdict
import csv
import jsonlines

import sys
from pathlib import Path

sys.path.append(str(Path('.').absolute()))
from concite.embedding.node2vec_wrapper import Node2VecEmb

class CitGraph(nx.MultiDiGraph):
#class CitGraph(nx.Graph):

    def load_edges(self, edge_path):
        with jsonlines.open(edge_path) as reader:
            self.add_edges_from([(ex['citing_paper_id'], ex['cited_paper_id'])
                for ex in reader])

    def random_edge_chunk(self):
        l1 = list(self.nodes())
        l2 = list(self.nodes())
        random.shuffle(l1)
        random.shuffle(l2)
        return set(zip(l1, l2))

    def filter_by_degree(self, degree_cutoff=2):
        node_count = self.number_of_nodes()
        while True:
            remove = [n for n, d in self.degree() if d < degree_cutoff]
            self.remove_nodes_from(remove)
            if node_count == self.number_of_nodes():
                break
            else:
                node_count = self.number_of_nodes()

    def negative_edges(self, n):
        negative_edges = set()
        edge_set = set(self.edges())
        while len(negative_edges) < n:
            negative_edges = negative_edges | self.random_edge_chunk().difference(edge_set)
        return list(negative_edges)[:n]

    def train_test_split_edges(self, test_ratio=0.2):
        # Split into train and test segments, while ensuring that at least
        # one edge is preserved for each node.
        edge_sample = list(self.edges())
        test_count = int(len(edge_sample) * test_ratio)
        random.shuffle(edge_sample)
        test_edges = []
        train_edges = []
        protected_edges = set([(node, random.choice(list(self.neighbors(node))))
            for node in self.nodes()])
        protected_edges = protected_edges.union(set([(e[1], e[0])
            for e in protected_edges]))
        for i, edge in enumerate(edge_sample):
            if len(test_edges) < test_count:
                if edge not in protected_edges:
                    test_edges.append(edge)
                else:
                    train_edges.append(edge)
            else:
                train_edges += edge_sample[i:]
                break
        return train_edges, test_edges

    def embed_edges(self, edges, l=40, d=128, p=0.5, q=0.5, name='emb', use_cache=True):
        return Node2VecEmb(edges, l, d, p, q, name=name, use_cache=use_cache)

    def load_node_data(self, path):
        node_set = set(self.nodes())
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row[0] in node_set:
                    for i, key in enumerate(['year', 'title', 'authors']):
                        self.nodes[row[0]][key] = row[i+1]
