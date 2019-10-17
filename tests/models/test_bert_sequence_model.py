from allennlp.common.testing import ModelTestCase
from allennlp.common.util import ensure_list

import sys

from pathlib import Path

sys.path.append(str(Path('.').absolute()))

from concite.models.bert_sequence_model import BertSequenceModel
from concite.dataset_readers.aclarc_trace_data_reader import AclarcTraceDatasetReader
from concite.modules.token_embedders.mixed_embedder import MixedEmbedder

class BertSequenceModelTest(ModelTestCase):
    
    def setUp(self):
        super(BertSequenceModelTest, self).setUp()
        self.set_up_model('tests/fixtures/acl_bert_trace_model.json',
                          'tests/fixtures/train_acl_citation_sequences_raw.txt')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
