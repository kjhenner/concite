import numpy as np
from typing import Dict, List, Union
from overrides import overrides
from collections import defaultdict
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

@DatasetReader.register("aclarc_dataset_reader")
class AclarcDocDatasetReader(DatasetReader):
    """
    Reads a file containing JSON-line formatted ACL-ARC document data creates
    a corresponding dataset.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 venues: Union[List[str], str] = 'all',
                 top_n_workshops: int = 10,
                 workshop_path: str = None,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BertBasicWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._venues = venues
        if self._venues == 'workshops':
            self._workshop_lookup = {}
            workshop_counts = defaultdict(int)
            with open(workshop_path) as f:
                for line in f.readlines():
                    items = line.split('\t')
                    workshop_counts[items[1]] += len(items[2:])
                    for paper_id in items[2:]:
                        self._workshop_lookup[paper_id] = items[1]
            workshop_counts = list(workshop_counts.items())
            workshop_counts.sort(key=lambda x: -x[1])
            self._top_workshops = [w[0] for w in workshop_counts[:top_n_workshops]]
            print(self.top_workshops)

    @overrides
    def _read(self, file_path):
        for ex in jsonlines.open(file_path):
            if self._venues == 'workshops':
                label = self._workshop_lookup.get(ex['paper_id'][:6])
                if label and label in self._top_workshops:
                    yield self.text_to_instance(
                        abstract=ex['abstract'],
                        label=label,
                        paper_id=ex['paper_id']
                    )
            elif self._venues == 'all' or ex['venue'] in self._venues:
                yield self.text_to_instance(
                    abstract=ex['abstract'],
                    label=ex['venue'],
                    paper_id=ex['paper_id']
                )

    @overrides
    def text_to_instance(self,
                abstract: str,
                label: str,
                paper_id: str) -> Instance:

        abstract_tokens = self._tokenizer.split_words(abstract)

        fields = {
            'abstract': TextField(abstract_tokens, self._token_indexers),
            'label': LabelField(label),
            'paper_id': LabelField(paper_id, label_namespace='paper_id_labels')
        }

        return Instance(fields)
