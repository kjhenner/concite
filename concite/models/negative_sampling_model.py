
from typing import Dict, Optional, List

import allennlp
import numpy as np
import torch
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides

@Model.register("negative_sampling_model")
class NegativeSamplingModel(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 verbose_metrics: False,
                 embedding_dim: int = 128,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(NegativeSamplingModel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.dropout = torch.nn.Dropout(dropout)

        self.verbose_metrics = verbose_metrics

        initializer(self)

    def _compute_loss(self,
                      embedded: torch.Tensor) -> torch.Tensor:

        loss = 0
        for inst in embedded:
            loss += torch.log(torch.sigmoid(torch.dot(inst[0,0,:], inst[0,1,:])))
            for neg_inst in inst[1:]:
                loss += torch.log(torch.sigmoid(torch.dot(-neg_inst[0,:], neg_inst[1,:])))
        return -loss / embedded.size[0]

    @overrides
    def forward(self,
                sequence: List[Dict[str, torch.LongTensor]]) -> Dict[str, torch.Tensor]:

        output_dict = {}
        embedded = self.text_field_embedder(sequence)

        loss = self._compute_loss(embedded)
        output_dict["loss"] = loss

        return output_dict
