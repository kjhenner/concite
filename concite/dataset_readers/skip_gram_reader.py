import numpy as np
from typing import Dict, List
from overrides import overrides
from collections import defaultdict
import json
import jsonlines
import logging
from random import randrange

import allennlp
from allennlp.data import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("skip_gram_reader")
class SkipGramReader(DatasetReader):
    """
    Dataset reader for creating skip-grams from sequence data.
    """

    def __init__(self,
                 lazy: bool = False,
                 window_size: int = 5,
                 tokenizer: Tokenizer = None,
                 indexers: Dict[str, TokenIndexer] = None
                 ) -> None:
        super().__init__(lazy)
        self.window_size = window_size
        self._tokenizer = tokenizer or JustSpacesWordSplitter()
        self._indexers = indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self,
              file_path: str):
        with open(file_path) as f:
            for line in f.readlines():
                tokens = line.strip().split(' ')
                for i, token in enumerate(tokens):
                    source_token = LabelField(token,
                                              label_namespace='source_token')
                    for j in range(i - self.window_size, i + self.window_size + 1):
                        if j < 0 or i == j or j > len(tokens) - 1:
                            continue
                        else:
                            nbr_token = LabelField(tokens[j],
                                                   label_namespace='nbr_token')
                            fields = {
                                'source_token': source_token,
                                'nbr_token': nbr_token
                            }
                            yield Instance(fields)
