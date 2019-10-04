from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from itertools import chain
import numpy as np
import jsonlines
import csv
import sys
import os


def train_ngram_lm(bin_path, train_path, model_path, order='3', model='ukndiscount'):

    with NamedTemporaryFile(mode='w') as f:
        
        bo_path = f.name
        args = [ os.path.join(bin_path, 'ngram-count'),
                 '-text',
                 train_path,
                 '-' + model,
                 '-order',
                 order,
                 '-no-sos',
                 '-no-eos',
                 '-lm',
                 model_path
               ]
        p = Popen(args, stdout=PIPE)
        print(p.communicate()[0].decode('utf-8'))

def test_ngram_lm(bin_path, test_path, model_path, order='3'):

    with NamedTemporaryFile(mode='w') as f:
        
        bo_path = f.name
        args = [ os.path.join(bin_path, 'ngram'),
                 '-ppl',
                 test_path,
                 '-order',
                 order,
                 '-no-sos',
                 '-no-eos',
                 '-lm',
                 model_path
               ]
        p = Popen(args, stdout=PIPE)
        print(p.communicate()[0].decode('utf-8'))

if __name__ == "__main__":
    bin_path = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]

    with NamedTemporaryFile(mode='w') as f:
        model_path = f.name

        train_ngram_lm(bin_path, train_path, model_path)
        test_ngram_lm(bin_path, test_path, model_path)
