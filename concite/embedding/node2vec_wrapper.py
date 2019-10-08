from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from itertools import chain
import numpy as np
import csv
import os
import sys

class Node2VecEmb:

    def __init__(self, edges, l, d, p, q, ow=False, verbose=True, name='emb', use_cache=True):

        self.verbose = verbose
        nodes = list(set([node for edge in edges for node in edge]))
        self.idx_to_node = dict(enumerate(nodes))
        self.node_to_idx = dict([(node, i) for i, node in enumerate(nodes)])
        ext = 'emb'
        if ow:
            ext = 'walks'
        out_path = os.path.abspath('{}_{}_{}_{}_{}.{}'.format(name, l, d, p, q, ext))

        if not os.path.exists(out_path) or not use_cache:
            with NamedTemporaryFile(mode='w') as f:
                if self.verbose:
                    print("Writing edge data for {} edges to {}...".format(len(edges), f.name))
                
                f.write('\n'.join(["{}\t{}".format(*map(self.node_to_idx.get, edge)) for edge in edges]))
                f.flush()
                
                edge_path = f.name
                args = ['/home/khenner/src/snap/examples/node2vec/node2vec',
                        '-i:{}'.format(edge_path),
                        '-o:{}'.format(out_path),
                        '-l:{}'.format(l),
                        '-d:{}'.format(d),
                        '-p:{}'.format(p),
                        '-q:{}'.format(q)]
                if ow:
                    args.append('-ow')
                p = Popen(args, stdout=PIPE)
                if self.verbose:
                    print(args)
                if self.verbose:
                    print(p.communicate()[0].decode('utf-8'))
        if ow:
            self.read_walks(out_path)
        else:
            self.read_embeddings(out_path)

    def edge_to_string(self, edge):
        return '{}\t{}'.format(*map(self.node_to_idx.get, edge))

    def to_vec(self, node):
        return self.array[self.node_to_vector_idx[node]]

    def to_hadamard_vec(self, edge):
        return self.to_vec(edge[0]) * self.to_vec(edge[1])

    def read_embeddings(self, path):
        self.array = []
        self.node_to_vector_idx = {}
        self.vector_idx_to_node = {}
        if self.verbose:
            print("Reading node embeddings...")
        with open(path) as f:
            for i, elems in enumerate(map(lambda x: x.split(), f.readlines()[1:])):
                z = np.array(list(map(float, elems[1:])))
                self.array.append(z)
                self.vector_idx_to_node[i] = self.idx_to_node[int(elems[0])]
                self.node_to_vector_idx[self.idx_to_node[int(elems[0])]] = i
        if self.verbose:
            print("Done reading node embeddings")
            print("Read {} embeddings".format(len(self.array)))

    def read_walks(self, path):
        self.walks = []
        if self.verbose:
            print("Reading walks...")
        with open(path) as f:
            self.walks = [[self.idx_to_node.get(int(idx)) for idx in line.split()] for line in f.readlines()]
        if self.verbose:
            print("Done reading walks")

    def write_embeddings(self, path):
        with open(path, 'w') as f:
            f.writelines(['{} {}\n'.format(node, ' '.join([str(i) for i in self.array[i]]))
                    for node, i, in self.node_to_vector_idx.items()])

if __name__ == '__main__':
    edge_file = sys.argv[1]
    out_path = sys.argv[2]

    with open(edge_file) as f:
        edges = [line.split()[:2] for line in f.readlines()]

    emb = Node2VecEmb(edges, 40, 128, 0.5, 0.5, use_cache=False)
    emb.write_embeddings(out_path)
