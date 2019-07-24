from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

import sys

from pathlib import Path

sys.path.append(str(Path('.').absolute()))

from dataset_readers.aclarc_data_reader import AclarcDocDatasetReader

class TestDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = AclarcDocDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/aclarc-train.jsonl'))
        assert isinstance(instances, list)
