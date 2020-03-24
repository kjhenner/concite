from allennlp.common.testing import ModelTestCase
from allennlp.common.util import ensure_list

import sys
import os

from pathlib import Path

sys.path.append(str(Path('.').absolute()))

from concite.models.acl_gat_classifier import AclGatClassifier
from concite.dataset_readers.acl_graph_reader import AclGraphReader

class AclGatClassifierTest(ModelTestCase):
    
    def setUp(self):
        super(AclGatClassifierTest, self).setUp()
        env_vars = {
                'SEED': '1',
                'PYTORCH_SEED': '1',
                'NUMPY_SEED': '1',
                'LABEL_FIELD': 'combined_workshop',
                'BERT_VOCAB': '/home/khenner/scibert_scivocab_uncased/vocab.txt',
                'BERT_WEIGHTS': '/home/khenner/scibert_scivocab_uncased/weights.tar.gz',
                'PRETRAINED_FILE': 'None',
                'CUDA_DEVICE': '-1',
                'EMBEDDING_DIM': '384',
                'HIDDEN_DIM': '100',
                'TOP_N': '10',
                'TRAINING_DATA': './tests/fixtures/train_10_combined_workshop_acl_data.jsonl',
                'DEV_DATA': './tests/fixtures/train_10_combined_workshop_acl_data.jsonl',
                'TEST_DATA': './tests/fixtures/train_10_combined_workshop_acl_data.jsonl',
                'EDGE_PATH': './tests/fixtures/acl_edge_data_new.jsonl',
                'USE_NODE_VECTOR': 'true',
                'USE_ABSTRACT': 'true'
        }
        os.environ.update(env_vars)
        self.set_up_model('tests/fixtures/test_gat_config.json',
                './tests/fixtures/train_10_combined_workshop_acl_data.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
