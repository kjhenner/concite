
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
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("pubmed_document_dataset_reader")
class PubmedDocumentDatasetReader(DatasetReader):
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
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with jsonlines.open(file_path) as reader:
            for ex in reader:
                yield self.text_to_instance(
                    #title=ex['title'],
                    abstract=ex['abstract'],
                    venue=ex['journal-id'],
                    graph_vector=np.array(ex['n2v_vector'])
                )

    @overrides
    def text_to_instance(self,
                #title: str,
                abstract: str,
                venue: str,
                graph_vector: np.ndarray) -> Instance:

        #title_tokens = self._tokenizer.tokenize(title)
        abstract_tokens = self._tokenizer.tokenize(abstract)

        fields = {
            #'title': TextField(title_tokens, self._token_indexers),
            'abstract': TextField(abstract_tokens, self._token_indexers),
            'venue': LabelField(venue),
            'graph_vector': ArrayField(graph_vector)
        }

        return Instance(fields)
