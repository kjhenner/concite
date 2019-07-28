import sys
import unittest

from pathlib import Path

sys.path.append(str(Path('.').absolute()))

from concite.dataset_preprocessors.citation_graph import CitGraph

class TestCitGraph(unittest.TestCase):

    def setUp(self):
        self.g = CitGraph()
        self.g.load_data_from_dir('tests/fixtures/cit_graph_test_data/')

    def test_vertex_and_edge_counts(self):
        self.assertEqual(len([v for v in self.g.vertices() if self.g.vp.abstract[v]]), 10)
        self.assertEqual(len([v for v in self.g.vertices() if not self.g.vp.abstract[v]]), 3)
        self.assertEqual(len(list(self.g.edges())), 23)

    def test_embedding(self):
        self.g.embed_edges(use_cache=False)
        self.assertEqual(
                len([v for v in self.g.vertices() if self.g.vp.graph_vector[v]]),
                self.g.num_vertices())
