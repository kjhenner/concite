import numpy as np
from typing import Dict, List
from overrides import overrides
import json
import jsonlines
import logging

import allennlp
from allennlp.data import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, ArrayField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
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
                 text_lookup_path: str,
                 embedded_text: str = 'title',
                 use_bos_eos: bool = True,
                 lazy: bool = False,
                 sent_len_limit: int = None,
                 sequence_limit: int = 15,
                 abstract_tokenizer: Tokenizer = None,
                 abstract_indexers: Dict[str, TokenIndexer] = None,
                 sequence_tokenizer: Tokenizer = None,
                 sequence_indexers: Dict[str, TokenIndexer] = None
                 ) -> None:
        super().__init__(lazy)
        if embedded_text == 'title':
            with open(text_lookup_path) as f:
                self.data_lookup = {
                    line[0]: {'abstract': line[2]}
                        for line in map(lambda x: x.split('\t'), f.readlines())
                }
        elif embedded_text == 'abstract':
            with jsonlines.open(text_lookup_path) as reader:
                self.data_lookup = {
                    item['paper_id']: item for item in reader
                }
        self.data_lookup['<s>'] = {'abstract':'[unused0]'}
        self.data_lookup['</s>'] = {'abstract':'[unused1]'}
        self._sent_len_limit = sent_len_limit
        self._abstract_tokenizer = abstract_tokenizer or BertBasicWordSplitter()
        self._abstract_indexers = abstract_indexers

        self._sequence_tokenizer = sequence_tokenizer or JustSpacesWordSplitter()
        self._sequence_indexers = sequence_indexers or {"tokens": SingleIdTokenIndexer()}


    @overrides
    def _read(self, file_path):
        with open(file_path) as f:
            for ex in f.readlines():
                abstracts = [self.data_lookup[paper_id].get('abstract')[:self._sent_len_limit] for paper_id in ex.split()]
                yield self.text_to_instance(
                    abstracts = abstracts,
                    trace_seq = ex,
                )

    @overrides
    def text_to_instance(self,
                abstracts: str,
                trace_seq: List[str]) -> Instance:
        
        # Joining the trace_seq back into a string makes it fit more easily
        # into the workflow.
        paper_ids = self._sequence_tokenizer.split_words(trace_seq)
        abstracts = [self._abstract_tokenizer.split_words(abstract) for abstract in abstracts]

        fields = {
            'paper_ids': TextField(paper_ids, self._sequence_indexers),
            'abstracts': ListField([TextField(abstract, self._abstract_indexers) for abstract in abstracts])
        }

        return Instance(fields)
