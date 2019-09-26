from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

import sys

from pathlib import Path

sys.path.append(str(Path('.').absolute()))

from concite.dataset_readers.neg_sampling_seq_reader import NegSamplingSeqReader

class TestNegSamplingSeqReader(AllenNlpTestCase):

    def test_read_from_walk_file(self):
        reader = NegSamplingSeqReader()
        instances = ensure_list(reader.read('tests/fixtures/acl_40_128_0.5_0.5.walks'))
        print(len(instances))
        assert isinstance(instances, list)
