from allennlp.common.testing import ModelTestCase
from allennlp.common.util import ensure_list

import sys

from pathlib import Path

sys.path.append(str(Path('.').absolute()))

from concite.models.negative_sampling_model import NegativeSamplingModel
from concite.dataset_readers.neg_sampling_seq_reader import NegSamplingSeqReader

class NegSamplingModelTest(ModelTestCase):
    
    def setUp(self):
        super(NegSamplingModelTest, self).setUp()
        self.set_up_model('tests/fixtures/neg_sampling_seq_model.json',
                          'tests/fixtures/acl_40_128_0.5_0.5.walks')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
