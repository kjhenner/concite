import numpy as np
from typing import Dict, List, Union
from overrides import overrides
from collections import defaultdict
import random
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

@DatasetReader.register("cocitation_reader")
class CocitationReader(DatasetReader):

    def __init__(self,
                 text_lookup_path: str,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BertBasicWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        with open(text_lookup_path) as f:
            self.data_lookup = {
                line[0]: line[2]
                    for line in map(lambda x: x.strip().split('\t'), f.readlines()) if len(line)>2
            }
        self.pid_set = list(self.data_lookup.keys())

    @overrides
    def _read(self, file_path):
        with open(file_path) as f:
            data = [line.split() for line in f.readlines()]
        pid_set = []
        for line in data:
            yield self.text_to_instance(
                pid_anchor = line[0],
                pid_pos = line[1],
            )

    @overrides
    def text_to_instance(self,
                pid_anchor: str,
                pid_pos: str) -> Instance:
        text_anchor = self.data_lookup[pid_anchor]
        tokens_anchor = self._tokenizer.split_words(text_anchor)
        text_pos = self.data_lookup[pid_pos]
        tokens_pos = self._tokenizer.split_words(text_pos)
        pid_neg = random.choice(self.pid_set)
        text_neg = self.data_lookup[pid_neg]
        tokens_neg = self._tokenizer.split_words(text_neg)

        fields = {
            'pid_anchor': LabelField(pid_anchor, label_namespace='paper_id_labels'),
            'text_anchor': TextField(tokens_anchor, self._token_indexers),
            'pid_pos': LabelField(pid_pos, label_namespace='paper_id_labels'),
            'text_pos': TextField(tokens_pos, self._token_indexers),
            'pid_neg': LabelField(pid_neg, label_namespace='paper_id_labels'),
            'text_neg': TextField(tokens_neg, self._token_indexers)
        }

        return Instance(fields)
