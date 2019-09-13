from typing import Dict, List, Tuple, Union, Optional

import torch
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.models.language_model import _SoftmaxLoss

# Borrowed from later release of AllenNLP
from concite.training.metrics import Perplexity

@Model.register('bert_sequence_model')
class BertSequenceModel(Model):
    
    def __init__(self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            contextualizer: Seq2SeqEncoder,
            num_samples: int = None,
            dropout: float = None) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder

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


        self._perplexity = Perplexity()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

    def _get_target_token_embeddings(self,
                                     token_embeddings: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
        zero_col = token_embeddings.new_zeros(mask.size(0), 1).to(dtype=torch.bool)
        shifted_mask = torch.cat([zero_col, mask[:, 0:-1]], dim=1)
        return token_embeddings.masked_select(shifted_mask.unsqueeze(-1)).view(-1, self._forward_dim)

    def _compute_loss(self,
                      lm_embeddings: torch.Tensor,
                      token_embeddings: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        mask = targets > 0

        non_masked_targets = targets.masked_select(mask) - 1

        non_masked_embeddings = lm_embeddings.masked_select(
                mask.unsqueeze(-1)).view(-1, self._forward_dim)

        return self._softmax_loss(non_masked_embeddings, non_masked_targets)

    def num_layers(self) -> int:
        return self_contextualizer.num_layers + 1

    def forward(self,
                abstracts: List[Dict[str, torch.LongTensor]],
                graph_vectors: List[np.ndarray],
                paper_ids: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the loss from the batch.
        """

        # Mask size is (batch size, sequence length, padded abstract length)
        mask = get_text_field_mask(abstracts, num_wrapping_dims=2)

        # Embed the abstracts and retrieve <CLS> tokens for each.
        embeddings = self._text_field_embedder(abstracts)[:, :, 0, :]

        contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
                embeddings, None
        )

        return_dict = {}

        assert isinstance(contextual_embeddings, torch.Tensor)

        print(paper_ids)

        # targets is like paper ids, but offset forward by 1 in the second
        # dimension.
        targets = torch.zeros_like(paper_ids)
        targets[:, 0:targets.size()[1] - 1] = paper_ids[:, 1:]

        print(targets)

        loss = self._compute_loss(contextual_embeddings, embeddings, targets)
        
        num_targets = torch.sum((targets > 0).long())
        if num_targets > 0:
            average_loss = loss / num_targets.float()
        else:
            average_loss = torch.tensor(0.0).to(targets.device)

        self._last_average_loss[0] = average_loss.detach().item()

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
            'noncontextual_embeddings': embeddings,
            'mask': mask
        })

        return return_dict
