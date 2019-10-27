from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

import sys

from pathlib import Path

sys.path.append(str(Path('.').absolute()))

from concite.dataset_readers.aclarc_data_reader import AclarcDocDatasetReader
from concite.dataset_readers.pubmed_document_dataset_reader import PubmedDocumentDatasetReader

class TestDatasetReader(AllenNlpTestCase):

#    def test_read_from_pubmed_file(self):
#        reader = PubmedDocumentDatasetReader()
#        instances = ensure_list(reader.read('tests/fixtures/n2v_pubmed_articles.jsonl'))
#        assert isinstance(instances, list)

    def test_read_from_aclarc_file(self):
        reader = AclarcDocDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/aclarc-train.jsonl'))
        assert isinstance(instances, list)
