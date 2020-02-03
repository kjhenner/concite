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
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides

@Model.register("cocitation_triplet")
class CocitationTriplet(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 node_embedder: TokenEmbedder,
                 verbose_metrics: False,
                 classifier_feedforward: FeedForward = None,
                 use_node_vector: bool = True,
                 use_abstract: bool = True,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(CocitationTriplet, self).__init__(vocab, regularizer)

        self.node_embedder = node_embedder
        self.text_field_embedder = text_field_embedder
        self.use_node_vector = use_node_vector
        self.use_abstract = use_abstract
        self.dropout = torch.nn.Dropout(dropout)
        self.num_classes = self.vocab.get_vocab_size("labels")

        self.verbose_metrics = verbose_metrics

        self.loss = torch.nn.TripletMarginLoss()

        initializer(self)

    @overrides
    def forward(self,
                text_anchor: Dict[str, torch.LongTensor],
                pid_anchor: torch.LongTensor,
                text_pos: Dict[str, torch.LongTensor],
                pid_pos: torch.LongTensor,
                text_neg: Dict[str, torch.LongTensor],
                pid_neg: torch.LongTensor) -> Dict[str, torch.Tensor]:

        if self.use_abstract and self.use_node_vector:
            anchor_embedding = torch.cat([
                self.text_field_embedder(text_anchor)[:, 0, :],
                self.node_embedder(pid_anchor)], dim=-1)
            pos_embedding = torch.cat([
                self.text_field_embedder(text_pos)[:, 0, :],
                self.node_embedder(pid_pos)], dim=-1)
            neg_embedding = torch.cat([
                self.text_field_embedder(text_neg)[:, 0, :],
                self.node_embedder(pid_neg)], dim=-1)
        elif self.use_abstract:
            anchor_embedding = self.text_field_embedder(text_anchor)[:, 0, :]
            pos_embedding = self.text_field_embedder(text_pos)[:, 0, :]
            neg_embedding = self.text_field_embedder(text_neg)[:, 0, :]
        else:
            anchor_embedding = self.node_embedder(pid_anchor)
            pos_embedding = self.node_embedder(pid_pos)
            neg_embedding = self.node_embedder(pid_neg)

        output_dict = {"loss": self.loss(anchor_embedding, pos_embedding, neg_embedding)}
        output_dict['embedding'] = anchor_embedding

        return output_dict
