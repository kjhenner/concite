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
    sequences. Return negative sampling examples with a specified ratio.
    """

    def __init__(self,
                 lazy: bool = False,
                 sequence_limit: int = 15,
                 negative_k: int = 15,
                 tokenizer: Tokenizer = None,
                 indexers: Dict[str, TokenIndexer] = None
                 ) -> None:
        super().__init__(lazy)
        self._split_size = 4
        self._negative_k = 10
        self._tokenizer = tokenizer or JustSpacesWordSplitter()
        self._indexers = indexers or {"tokens": SingleIdTokenIndexer()}

    def sample_tokens(self, token_dist, token_count):
        tokens = []
        for _ in range(self._negative_k):
            rand_idx = randrange(token_count)
            for k, v in token_dist.items():
                if v < rand_idx:
                    tokens += k
                    break
                else:
                    rand_idx -= v
        return tokens

    @overrides
    def _read(self, file_path):
        token_dist = defaultdict(float)
        with open(file_path) as f:
            for ex in f.readlines():
                for token in ex.split():
                    token_dist[token] += 1
        token_count = sum(token_dist.values())
        with open(file_path) as f:
            for ex in f.readlines():
                trace_seq = ex.split()
                u = trace_seq.pop(0)
                pos_pairs = [' '.join((u, v)) for v in trace_seq]
                for pos_pair in pos_pairs:
                    neg_pairs = [' '.join((pos_pair[0], v)) for v in self.sample_tokens(token_dist, token_count)]
                    yield self.text_to_instance(
                            pairs = [
                                self._tokenizer.split_words(pair)
                                for pair in [pos_pair] + neg_pairs
                            ]
                        )

    @overrides
    def text_to_instance(self,
                         pairs: List[str]) -> Instance:
        
        fields = {
            'sequence': ListField([
                TextField(pair, self._indexers)
                for pair in pairs
            ])
        }

        return Instance(fields)
