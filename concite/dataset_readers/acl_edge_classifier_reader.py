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
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("acl_edge_classifier_reader")
class AclEdgeClassifierReader(DatasetReader):
    """
    Reads a file containing JSON-line formatted ACL-ARC document data creates
    a corresponding dataset.
    """

    def __init__(self,
                 label_field: str,
                 text_field: str,
                 paper_lookup_path: str,
                 sent_max_len: int = 256,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BertBasicWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_field = label_field
        self._text_field = text_field
        self._sent_max_len = sent_max_len

        self._paper_lookup = self.load_paper_lookup(paper_lookup_path)

    def load_paper_lookup(self, path):
        with open(path) as f:
            return {ex['paper_id']: ex for ex in jsonlines.Reader(f)}

    @overrides
    def _read(self, file_path):
        for ex in jsonlines.open(file_path):
            citing_paper = self._paper_lookup[ex['metadata']['citing_paper']]
            cited_paper = self._paper_lookup[ex['metadata']['cited_paper']]
            yield self.text_to_instance(
                citing_paper_id=ex['metadata']['citing_paper'],
                cited_paper_id=ex['metadata']['cited_paper'],
                citing_text=citing_paper[self._text_field],
                cited_text=cited_paper[self._text_field],
                label=ex['metadata'][self._label_field],
            )

    @overrides
    def text_to_instance(self,
                citing_paper_id: str,
                cited_paper_id: str,
                citing_text: str,
                cited_text: str,
                label: str) -> Instance:

        citing_tokens = self._tokenizer.split_words(citing_text)[:self._sent_max_len]
        cited_tokens = self._tokenizer.split_words(cited_text)[:self._sent_max_len]

        fields = {
            'text': TextField(
                citing_tokens + [Token("[SEP]")] + cited_tokens,
                self._token_indexers
            ),
            'label': LabelField(label),
            'citing_paper_id': LabelField(citing_paper_id, label_namespace='paper_id_labels'),
            'cited_paper_id': LabelField(cited_paper_id, label_namespace='paper_id_labels')
        }

        return Instance(fields)
