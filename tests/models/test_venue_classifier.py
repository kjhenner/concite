from allennlp.common.testing import ModelTestCase
from allennlp.common.util import ensure_list

import sys

from pathlib import Path

sys.path.append(str(Path('.').absolute()))

from concite.models.venue_classifier import VenueClassifier
from concite.dataset_readers.pubmed_document_dataset_reader import PubmedDocumentDatasetReader

class VenueClassifierTest(ModelTestCase):
    
    def setUp(self):
        super(VenueClassifierTest, self).setUp()
        self.set_up_model('tests/fixtures/test_predict_venue.json',
                          'tests/fixtures/n2v_pubmed_articles.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


