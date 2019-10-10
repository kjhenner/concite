
from typing import Dict, Optional, List

import allennlp
import numpy as np
import torch
import torch.nn.functional as F
import math
from allennlp.common import Params
from allennlp.data import Instance
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, TokenEmbedder, Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides

@Model.register("negative_sampling_model")
class NegativeSamplingModel(Model):
    def __init__(self, vocab: Vocabulary,
                 verbose_metrics: False,
                 embedding_dim: int = 128,
                 dropout: float = 0.2,
                 neg_samples: int = 10,
                 cuda_device: int = 7,
                 pretrained_file: str = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(NegativeSamplingModel, self).__init__(vocab, regularizer)

        self.embedder = Embedding(num_embeddings=vocab.get_vocab_size('source_token'),
                                   embedding_dim=embedding_dim,
                                   pretrained_file=pretrained_file)
        self.neg_samples = neg_samples
        self.cuda_device = cuda_device
        self.dropout = torch.nn.Dropout(dropout)
        self.verbose_metrics = verbose_metrics

        # Compute negative sampling probabilities
        # Based on https://github.com/mhagiwara/realworldnlp
        token_probs = {}
        token_counts = vocab._retained_counter['source_token']
        total_counts = float(sum(token_counts.values()))
        total_probs = 0.
        for token, counts in token_counts.items():
            adjusted_freq = math.pow(counts / total_counts, 0.75)
            token_probs[token] = adjusted_freq
            total_probs += adjusted_freq

        self.neg_sample_probs = np.ndarray((vocab.get_vocab_size('source_token'),))
        for idx, token in vocab.get_index_to_token_vocabulary('source_token').items():
            self.neg_sample_probs[idx] = token_probs.get(token, 0) / total_probs

        initializer(self)

    def _compute_loss(self,
                      pair: List[torch.Tensor]) -> torch.Tensor:

        loss = 0
        for inst in embedded:
            loss += torch.log(torch.sigmoid(torch.dot(inst[0,0,:], inst[0,1,:])))
            for neg_inst in inst[1:]:
                loss += torch.log(torch.sigmoid(torch.dot(-neg_inst[0,:], neg_inst[1,:])))
        return -loss / embedded.size[0]

    @overrides
    def forward(self,
                source_token: Dict[str, torch.LongTensor],
                nbr_token: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        batch_size = source_token.shape[0]

        
        embedded_source = self.embedder(source_token)
        embedded_nbr = self.embedder(nbr_token)

        log_prob = F.logsigmoid(torch.mul(embedded_source, embedded_nbr).sum(dim=1))

        negative_nbr = np.random.choice(a=self.vocab.get_vocab_size('source_token'),
                                        size=batch_size * self.neg_samples,
                                        p=self.neg_sample_probs)
        negative_nbr = torch.LongTensor(negative_nbr).view(batch_size, self.neg_samples)
        if self.cuda_device > -1:
            negative_nbr = negative_nbr.to(self.cuda_device)

        embedded_neg_nbr = self.embedder(negative_nbr)
        negative = torch.bmm(embedded_neg_nbr, embedded_source.unsqueeze(2)).squeeze()

        log_prob += F.logsigmoid(-1. * negative).sum(dim=1)

        return {'loss': -log_prob.sum() / batch_size}
