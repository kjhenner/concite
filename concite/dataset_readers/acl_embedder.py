import numpy as np
from typing import Dict, List, Union
from overrides import overrides
from collections import defaultdict
import json
import jsonlines
import logging
from random import choice

import allennlp
from allennlp.data import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("acl_embedder_reader")
class AclEmbedderReader(DatasetReader):
    """
    Reads a file containing JSON-line formatted ACL-ARC document data creates
    a corresponding dataset.
    """

    def __init__(self,
                 text_field: str,
                 paper_lookup_path: str,
                 sent_max_len: int = 256,
                 neg_samples: int = 5,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BertBasicWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._text_field = text_field
        self._sent_max_len = sent_max_len
        self._neg_samples = neg_samples

        self._paper_lookup = self.load_paper_lookup(paper_lookup_path)
        self._paper_ids = list(self._paper_lookup.keys())

    def load_paper_lookup(self, path):
        with open(path) as f:
            return {ex['paper_id']: ex for ex in jsonlines.Reader(f)}

    @overrides
    def _read(self, file_path):
        with open(file_path) as f:
            examples = [line.strip().split() for line in f.readlines()]
        for ex in examples:
            paper_a=self._paper_lookup[ex[0]]
            paper_b=self._paper_lookup[ex[1]]
            yield self.text_to_instance(
                paper_a=ex[0],
                paper_b=ex[1],
                text_a=paper_a[self._text_field],
                text_b=paper_b[self._text_field],
                label=1,
            )
            for _ in range(self._neg_samples):
                paper_a = self._paper_lookup[choice(self._paper_ids)]
                paper_b = self._paper_lookup[choice(self._paper_ids)]
                yield self.text_to_instance(
                    paper_a = paper_a['paper_id'],
                    paper_b = paper_b['paper_id'],
                    text_a = paper_a[self._text_field],
                    text_b = paper_b[self._text_field],
                    label=-1,
                )

    @overrides
    def text_to_instance(self,
                paper_a: str,
                paper_b: str,
                text_a: str,
                text_b: str,
                label: int) -> Instance:

        tokens_a = self._tokenizer.split_words(text_a)[:self._sent_max_len]
        tokens_b = self._tokenizer.split_words(text_b)[:self._sent_max_len]

        fields = {
            'label': LabelField(label, skip_indexing=True),
            'text_a': TextField(tokens_a, self._token_indexers),
            'text_b': TextField(tokens_a, self._token_indexers),
            'paper_a': LabelField(paper_a, label_namespace='paper_id_labels'),
            'paper_b': LabelField(paper_b, label_namespace='paper_id_labels')
        }

        return Instance(fields)
