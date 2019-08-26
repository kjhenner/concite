
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

@DatasetReader.register("aclarc_dataset_reader")
class AclarcDocDatasetReader(DatasetReader):
    """
    Reads a file containing JSON-line formatted ACL-ARC document data creates
    a corresponding dataset.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BertBasicWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        for ex in jsonlines.open(file_path):
            yield self.text_to_instance(
                abstract=ex['abstract'],
                label=ex['venue'],
                graph_vector=np.array(ex['graph_vector'])
            )

    @overrides
    def text_to_instance(self,
                abstract: str,
                label: str,
                graph_vector: np.ndarray) -> Instance:

        abstract_tokens = self._tokenizer.split_words(abstract)

        fields = {
            'abstract': TextField(abstract_tokens, self._token_indexers),
            'label': LabelField(label),
            'graph_vector': ArrayField(graph_vector)
        }

        return Instance(fields)
