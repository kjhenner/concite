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
from allennlp.data.fields import TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("neg_sampling_seq_reader")
class NegSamplingSeqReader(DatasetReader):
    """
    Reads in a text file consisting of N2V random walks, trace data, or other
    sequences.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 indexers: Dict[str, TokenIndexer] = None
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or JustSpacesWordSplitter()
        self._indexers = indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(file_path) as f:
            for ex in f.readlines():
                yield self.text_to_instance(
                        walk = self._tokenizer.split_words(ex))

    @overrides
    def text_to_instance(self,
                         walk: List[str]) -> Instance:
        fields = {
            'walk': TextField(walk, self._indexers)
        }
        return Instance(fields)
