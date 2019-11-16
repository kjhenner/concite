from typing import Dict, Optional

import allennlp
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import jsonlines
from allennlp.common import Params
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from overrides import overrides

@Model.register("acl_embedder")
class AclEmbedder(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 node_embedder: TokenEmbedder,
                 verbose_metrics: False,
                 use_node_vector: bool = True,
                 use_text: bool = True,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AclEmbedder, self).__init__(vocab, regularizer)

        self.node_embedder = node_embedder
        self.text_field_embedder = text_field_embedder
        self.use_node_vector = use_node_vector
        self.use_text = use_text
        self.dropout = torch.nn.Dropout(dropout)
        self.verbose_metrics = verbose_metrics

        self.loss = torch.nn.CosineEmbeddingLoss()
        initializer(self)

    @overrides
    def forward(self,
                text_a: Dict[str, torch.LongTensor],
                text_b: Dict[str, torch.LongTensor],
                paper_a: torch.LongTensor,
                paper_b: torch.LongTensor,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        if self.use_text and self.use_node_vector:
            emb_text_a = self.text_field_embedder(text_a)[:, 0, :]
            emb_text_b = self.text_field_embedder(text_b)[:, 0, :]
            emb_node_a = self.node_embedder(paper_a)
            emb_node_b = self.node_embedder(paper_b)
            emb_a = self.dropout(torch.cat([embedded_text_a, emb_node_a], dim=-1))
            emb_b = self.dropout(torch.cat([embedded_text_b, emb_node_b], dim=-1))
        elif self.use_text:
            emb_a = self.dropout(self.text_field_embedder(text_a)[:, 0, :])
            emb_b = self.dropout(self.text_field_embedder(text_b)[:, 0, :])
        elif self.use_node_vector:
            emb_a = self.dropout(self.node_embedder(paper_a))
            emb_b = self.dropout(self.node_embedder(paper_b))

        if label is not None:
            loss = self.loss(emb_a,emb_b,label.float())
            output_dict = {"loss": loss}

        return output_dict
