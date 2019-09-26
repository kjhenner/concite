
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
        with jsonlines.open(abstract_lookup_path) as reader:
            self.data_lookup = {
                item['paper_id']: item for item in reader
            }
        self._use_bos_eos = use_bos_eos
        self._sent_len_limit = sent_len_limit
        self._sequence_limit = sequence_limit
        self._split_size = 4
        self._abstract_tokenizer = abstract_tokenizer or BertBasicWordSplitter()
        self._abstract_indexers = abstract_indexers

        self._sequence_tokenizer = sequence_tokenizer or JustSpacesWordSplitter()
        self._sequence_indexers = sequence_indexers or {"tokens": SingleIdTokenIndexer()}

    def get_statistics(self, file_path):
        stats = {
            'raw': {
                'total_sequences': 0,
                'total_items': 0,
                'max_length': 0
            },
            'skip_filtered': {
                'total_sequences': 0,
                'total_items': 0,
                'max_length': 0
            },
            'hop_filtered': {
                'total_sequences': 0,
                'total_items': 0,
                'max_length': 0
            }
        }
        with open(file_path) as f:
            for ex in f.readlines():
                pids = ex.split()
                cnt = len(pids)
                stats['raw']['total_sequences'] += 1
                stats['raw']['total_items'] += cnt
                stats['raw']['max_length'] = max(stats['raw']['max_length'], cnt)
                if all([self.data_lookup.get(pid) and self.data_lookup[pid].get('abstract') for pid in pids]):
                    stats['skip_filtered']['total_sequences'] += 1
                    stats['skip_filtered']['total_items'] += cnt
                    stats['skip_filtered']['max_length'] = max(stats['raw']['max_length'], cnt)
                pids = [pid for pid in pids if self.data_lookup.get(pid) and self.data_lookup.get(pid).get('abstract')]
                if len(pids) > 1:
                    cnt = len(pids)
                    stats['hop_filtered']['total_sequences'] += 1
                    stats['hop_filtered']['total_items'] += cnt
                    stats['hop_filtered']['max_length'] = max(stats['raw']['max_length'], cnt)
        stats['raw']['mean'] = float(stats['raw']['total_items'] / stats['raw']['total_sequences'])
        print(stats)
        stats['skip_filtered']['mean'] = float(stats['skip_filtered']['total_items'] / stats['skip_filtered']['total_sequences'])
        stats['hop_filtered']['mean'] = float(stats['hop_filtered']['total_items'] / stats['hop_filtered']['total_sequences'])
        return stats

    @overrides
    def _read(self, file_path):
        #print(self.get_statistics(file_path))
        with open(file_path) as f:
            for ex in f.readlines():
                trace_seq = ex.split()
                if len(trace_seq) > self._sequence_limit:
                    split_seqs = zip(*[iter(trace_seq)]*self._split_size)
                else:
                    split_seqs = [trace_seq]
                for split_seq in split_seqs:
                    # For now, just skip papers outside of dataset intersection
                    # and those without abstracts.
                    if len(split_seq) > 1 and all([self.data_lookup.get(paper_id) and self.data_lookup[paper_id].get('abstract') for paper_id in split_seq]):
                        abstracts = [self.data_lookup[paper_id].get('abstract')[:self._sent_len_limit] for paper_id in split_seq]
                        if self._use_bos_eos:
                            split_seq = ["<BOS>", *split_seq, "<EOS>"]
                        yield self.text_to_instance(
                            abstracts = ["[unused0]", *abstracts, "[unused1]"],
                            trace_seq = split_seq,
                        )

    @overrides
    def text_to_instance(self,
                abstracts: List[str],
                trace_seq: List[str]) -> Instance:
        
        # Joining the trace_seq back into a string makes it fit more easily
        # into the workflow.
        paper_ids = self._sequence_tokenizer.split_words(' '.join(trace_seq))
        abstracts = [self._abstract_tokenizer.split_words(abstract) for abstract in abstracts]

        fields = {
            'paper_ids': TextField(paper_ids, self._sequence_indexers),
            'abstracts': ListField([TextField(abstract, self._abstract_indexers) for abstract in abstracts])
        }

        return Instance(fields)
