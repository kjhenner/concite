
import numpy as np
from typing import Dict, List
from overrides import overrides
import json
import jsonlines
import logging

import allennlp
from allennlp.data import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
#from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("aclarc_trace_dataset_reader")
class AclarcTraceDatasetReader(DatasetReader):
    """
    Reads a file containing ACL-ARC trace data and creates a corresponding
    dataset. Each trace is handled as if it were a sentence, where each visited
    paper is a token in that sentence.
    """

    def __init__(self,
                 lazy: bool = False,
                 ) -> None:
        super().__init__(lazy)
        self._seq_token_indexers = {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = WordTokenizer(JustSpacesWordSplitter())

    @overrides
    def _read(self, file_path):
        with open(file_path) as f:
            for ex in f.readlines():
                # We're only interested in sequences, so remove
                # single paper sessions.
                if len(ex.split()) > 1:
                    yield self.text_to_instance(
                            trace_seq = ex
                    )

    @overrides
    def text_to_instance(self,
                trace_seq: str) -> Instance:

        trace_tokens = self._tokenizer.tokenize(trace_seq)

        # By treating the trace sequence as a TextField, we can take
        # advantage of language modeling facilities for our sequence
        # modeling.
        fields = {
            'source': TextField(trace_tokens, self._seq_token_indexers)
        }

        return Instance(fields)
