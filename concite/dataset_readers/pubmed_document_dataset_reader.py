
import numpy as np
from typing import Dict, List
from overrides import overrides
import json
import jsonlines
import logging

import allennlp
from allennlp.data import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("pubmed_document_dataset_reader")
class PubmedDocumentDatasetReader(DatasetReader):
    """
    Reads a file containing JSON-line formatted ACL-ARC document data creates
    a corresponding dataset.
    """

    def __init__(self,
                 max_instances: int = None,
                 min_abstract_len: int = 10,
                 min_jid_count: int = 50,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy)
        self.max_instances = max_instances
        self.min_abstract_len = min_abstract_len
        self.min_jid_count = min_jid_count
        self._tokenizer = tokenizer or BertBasicWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with jsonlines.open(file_path) as reader:
            count = 0
            for ex in reader:
                if self.max_instances and count > self.max_instances:
                    break
                if ex.get('abstract') and len(ex['abstract']) >= self.min_abstract_len and ex.get('journal-id') and ex.get('jid_count') and ex['jid_count'] >= self.min_jid_count:
                    count += 1
                    yield self.text_to_instance(
                        #title=ex['title'],
                        abstract=ex['abstract'],
                        label=ex['journal-id'],
                        graph_vector=np.array(ex['n2v_vector'])
                    )

    @overrides
    def text_to_instance(self,
                #title: str,
                abstract: str,
                label: str,
                graph_vector: np.ndarray) -> Instance:

        #title_tokens = self._tokenizer.tokenize(title)
        abstract_tokens = self._tokenizer.split_words(abstract)

        fields = {
            #'title': TextField(title_tokens, self._token_indexers),
            'abstract': TextField(abstract_tokens, self._token_indexers),
            'label': LabelField(label),
            'graph_vector': ArrayField(graph_vector)
        }

        return Instance(fields)
