from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from itertools import chain
import numpy as np
import csv
import os
import sys

class Node2VecEmb:

    def __init__(self, edges, l, d, p, q, w=False, ow=False, verbose=True):
        self.verbose = verbose
        nodes = list(set([node for edge in edges for node in edge]))
        self.idx_to_node = dict(enumerate(nodes))
        self.node_to_idx = dict([(node, i) for i, node in enumerate(nodes)])
        if ow:
            ext = 'walks'
        out_path = os.path.abspath('{}_{}_{}_{}.emb'.format(l, d, p, q))
        with NamedTemporaryFile(mode='w') as f:
            if self.verbose:
                print("Writing {} edges to {}".format(len(edges), f.name))
            f.write('\n'.join([self.edge_to_string(edge) for edge in edges]))
            f.flush()
            edge_path = f.name
            self.run_n2v(edge_path, out_path, l, d, p, q, ow, w)
        if ow:
            self.read_walks(out_path)
        else:
            self.read_embeddings(out_path)

    def run_n2v(self, edge_path, out_path, l, d, p, q, ow, w):
        args = ['/home/khenner/src/snap/examples/node2vec/node2vec',
                '-i:{}'.format(edge_path),
                '-o:{}'.format(out_path),
                '-l:{}'.format(l),
                '-d:{}'.format(d),
                '-p:{}'.format(p),
                '-e:2',
                '-q:{}'.format(q)]
        if ow:
            args.append('-ow')
        if w:
            args.append('-w')
        p = Popen(args, stdout=PIPE)
        if self.verbose:
            print(args)
            for line in iter(p.stdout.readline, b''):
                sys.stdout.write(line.decode('utf-8'))
            output = p.communicate()[0]
            print(output.decode('utf-8'))

    def edge_to_string(self, edge):
        return '{}\t{}\t{}'.format(*map(self.node_to_idx.get, edge))

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
        edges = [line.split() for line in f.readlines()]

    emb = Node2VecEmb(edges, 40, 128, 0.5, 0.5)
    emb.write_embeddings(out_path)
