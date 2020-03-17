import sys
import jsonlines
import torch
import numpy as np
import scipy.sparse as sp
import logging
from overrides import overrides

from typing import Dict

import allennlp
from allennlp.data import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("acl_graph_reader")
class AclGraphReader(DatasetReader):

    def __init__(self,
                 label_field: str,
                 text_field: str,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy)

        self._tokenizer = tokenizer or BertBasicWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_field = label_field
        self._text_field = text_field

    @overrides
    def _read(self, file_path):
        for ex in jsonlines.open(file_path):
            yield self.text_to_instance(
                text=ex[self._text_field],
                label=ex[self._label_field],
                paper_id=ex['paper_id']
            )

    @overrides
    def text_to_instance(self,
            text: str,
            label: str,
            paper_id: str) -> Instance:
        try:
            text_tokens = self._tokenizer.split_words(text)
        except AttributeError:
            text_tokens = self._tokenizer.tokenize(text)

        fields = {
            'text': TextField(text_tokens, self._token_indexers),
            'label': LabelField(label),
            'paper_id': LabelField(paper_id, label_namespace='paper_id_labels')
        }
        return Instance(fields)
