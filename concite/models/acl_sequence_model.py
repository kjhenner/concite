from typing import Dict, List, Tuple, Union, Optional
from overrides import overrides
from collections import defaultdict

import torch
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn import util
from allennlp.nn.util import get_text_field_mask
#from allennlp.models.language_model import _SoftmaxLoss
from concite.modules.token_embedders.mixed_embedder import MixedEmbedder

from concite.training.metrics import *
# Borrowed from later release of AllenNLP
from concite.training.metrics import Perplexity


class _SoftmaxLoss(torch.nn.Module):
    """
    Given some embeddings and some targets, applies a linear layer
    to create logits over possible words and then returns the
    negative log likelihood.
    """

    def __init__(self,
                 num_words: int,
                 embedding_dim: int) -> None:
        super().__init__()

        self.softmax_w = torch.nn.Parameter(
            torch.randn(embedding_dim, num_words) / np.sqrt(embedding_dim)
        )
        self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

    def forward(self,
                embeddings: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:

        # embeddings is size (n, embedding_dim)
        # targets is (batch_size, ) with the correct class id
        # Does not do any count normalization / divide by batch size
        probs = torch.nn.functional.log_softmax(torch.matmul(embeddings, self.softmax_w) + self.softmax_b, dim=-1)

        return torch.nn.functional.nll_loss(probs, targets.long(), reduction="sum")

    def probs(self,
              embeddings: torch.Tensor) -> torch.Tensor:

        return torch.nn.functional.log_softmax(torch.matmul(embeddings, self.softmax_w) + self.softmax_b, dim=1)

@Model.register('acl_sequence_model')
class AclSequenceModel(Model):
    
    def __init__(self,
            vocab: Vocabulary,
            seq_embedder: TextFieldEmbedder,
            abstract_text_field_embedder: TextFieldEmbedder,
            contextualizer: Seq2SeqEncoder,
            use_abstracts: bool = True,
            use_node_vectors: bool = True,
            num_samples: int = None,
            dropout: float = None) -> None:
        super().__init__(vocab)

        self._abstract_text_field_embedder = abstract_text_field_embedder

        self._use_abstracts = use_abstracts

        self._use_node_vectors = use_node_vectors

        self._seq_embedder = seq_embedder

        # lstm encoder uses PytorchSeq2SeqWrapper for pytorch lstm
        self._contextualizer = contextualizer

        self._forward_dim = contextualizer.get_output_dim()

        if num_samples is not None:
            self._softmax_loss = SampledSoftmaxLoss(num_words=vocab.get_vocab_size(),
                                                    embedding_dim=self._forward_dim,
                                                    num_samples=num_samples,
                                                    sparse=False)
        else:
            self._softmax_loss = _SoftmaxLoss(num_words=vocab.get_vocab_size(),
                                              embedding_dim=self._forward_dim)

        self._n_list = range(1, 50)
        self._recall_at_n = {}
        for n in self._n_list:
            self._recall_at_n[n] = RecallAtN(n)
        self._perplexity = Perplexity()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def _compute_loss(self,
                      lm_embeddings: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:

        # Because the targets are offset by 1, we re-mask to
        # remove the final 0 in the targets
        mask = targets > 0
        non_masked_targets = targets.masked_select(mask) - 1
        non_masked_embeddings = lm_embeddings.masked_select(
                mask.unsqueeze(-1)).view(-1, self._forward_dim)

        return self._softmax_loss(non_masked_embeddings, non_masked_targets)

    def _compute_probs(self,
                      lm_embeddings: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:

        # Because the targets are offset by 1, we re-mask to
        # remove the final 0 in the targets
        mask = targets > 0
        non_masked_targets = targets.masked_select(mask) - 1
        non_masked_embeddings = lm_embeddings.masked_select(
                mask.unsqueeze(-1)).view(-1, self._forward_dim)

        return self._softmax_loss.probs(non_masked_embeddings)

    def num_layers(self) -> int:
        return self_contextualizer.num_layers + 1

    def forward(self,
                abstracts: Dict[str, torch.LongTensor],
                paper_ids: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the loss from the batch.
        """

        if self._use_abstracts and self._use_node_vectors:
            embeddings = torch.cat([
                self._abstract_text_field_embedder(abstracts)[:, :, 0, :],
                self._seq_embedder(paper_ids)], dim = -1)
            mask = get_text_field_mask(abstracts, num_wrapping_dims=1)
            mask = mask.sum(dim=-1) > 0
        elif self._use_abstracts:
            embeddings = self._abstract_text_field_embedder(abstracts)[:, :, 0, :]
            mask = get_text_field_mask(abstracts, num_wrapping_dims=1)
            mask = mask.sum(dim=-1) > 0
        elif self._use_node_vectors:
            embeddings = self._seq_embedder(paper_ids)
            mask = get_text_field_mask(paper_ids)
        contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
                embeddings, mask.long()
        )

        contextual_embeddings_with_dropout = self._dropout(contextual_embeddings)

        return_dict = {}

        assert isinstance(contextual_embeddings_with_dropout, torch.Tensor)

        # targets is like paper ids, but offset forward by 1 in the second
        # dimension.
        targets = torch.zeros_like(paper_ids['tokens'])
        targets[:, 0:targets.size()[1] - 1] = paper_ids['tokens'][:, 1:]

        loss = self._compute_loss(contextual_embeddings_with_dropout, targets)
        
        num_targets = torch.sum((targets > 0).long())
        if num_targets > 0:
            average_loss = loss / num_targets.float()
        else:
            average_loss = torch.tensor(0.0).to(targets.device)

        perplexity = self._perplexity(average_loss)

        if not self.training:
            self.get_recall_at_n(contextual_embeddings, targets)

        if num_targets > 0:
            return_dict.update({
                'loss': average_loss,
                'batch_weight': num_targets.float()
            })
        else:
            return_dict.update({
                'loss': average_loss
            })

        return_dict.update({
            'lm_embeddings': contextual_embeddings,
            'lm_targets': targets,
            'noncontextual_embeddings': embeddings,
        })

        return return_dict

    def get_recall_at_n(self, embeddings, targets):
        top_n = []
        for embeddings, targets in zip(embeddings.detach(), targets.detach()):
            probs = self._compute_probs(embeddings, targets)
            top_probs, top_indices = probs.topk(k=max(self._n_list), dim=-1)
            top_n.append([[self.vocab.get_token_from_index(int(i))
                for i in top_n]
                for top_n in top_indices])
            mask = targets > 0
            non_masked_targets = targets.masked_select(mask) - 1
            for n in self._n_list:
                self._recall_at_n[n](non_masked_targets, top_indices)
        return top_n

    def get_metrics(self, reset: bool = False):
        metrics = {"perplexity": self._perplexity.get_metric(reset=reset)}
        if not self.training:
            for n in self._n_list:
                recall = self._recall_at_n[n].get_metric(reset=reset)
                metrics.update({
                    "recall_at_{}".format(n) : recall
                })
        return metrics
