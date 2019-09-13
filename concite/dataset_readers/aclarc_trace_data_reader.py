
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
                 abstract_lookup_path: str,
                 lazy: bool = False,
                 abstract_tokenizer: Tokenizer = None,
                 abstract_indexers: Dict[str, TokenIndexer] = None,
                 sequence_tokenizer: Tokenizer = None,
                 sequence_indexers: Dict[str, TokenIndexer] = None
                 ) -> None:
        super().__init__(lazy)
        with jsonlines.open(abstract_lookup_path) as reader:
            self.data_lookup = {
                item['paper_id']: item for item in reader
            }
        self.sent_len_limit = 256
        self._abstract_tokenizer = abstract_tokenizer or BertBasicWordSplitter()
        self._abstract_indexers = abstract_indexers

        self._sequence_tokenizer = sequence_tokenizer or JustSpacesWordSplitter()
        self._sequence_indexers = sequence_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(file_path) as f:
            for ex in f.readlines():
                trace_seq = ex.split()
                # For now, just skip papers outside of dataset intersection.
                if all([self.data_lookup.get(paper_id) for paper_id in trace_seq]):
                    yield self.text_to_instance(
                        trace_seq = trace_seq,
                        abstracts = [
                            self.data_lookup.get(paper_id, {}).get('abstract')
                            for paper_id in trace_seq
                        ],
                        graph_vectors = [
                            np.array(self.data_lookup.get(paper_id, {}).get('graph_vector'))
                            for paper_id in trace_seq
                        ]
                    )

    @overrides
    def text_to_instance(self,
                trace_seq: List[str],
                abstracts: List[str],
                graph_vectors: List[np.ndarray]) -> Instance:
        
        abstracts = [self._abstract_tokenizer.split_words(abstract)[:self.sent_len_limit] for abstract in abstracts]

        paper_ids = self._sequence_tokenizer.tokenize(trace_seq)

        fields = {
                'abstracts': ListField([
		    TextField(abstract, self._abstract_indexers)
		    for abstract in abstracts
		]),
                'graph_vectors': ListField([
                    ArrayField(vec)
                    for vec in graph_vectors
                ]),
                'paper_ids': TextField(paper_ids, self._sequence_indexers)
        }

        return Instance(fields)
