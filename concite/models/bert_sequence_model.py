from typing import Dict, List, Tuple, Union, Optional
from overrides import overrides

import torch
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.models.language_model import _SoftmaxLoss
from concite.modules.token_embedders.mixed_embedder import MixedEmbedder

# Borrowed from later release of AllenNLP
from concite.training.metrics import Perplexity

@Model.register('bert_sequence_model')
class BertSequenceModel(Model):
    
    def __init__(self,
            vocab: Vocabulary,
            seq_embedder: TextFieldEmbedder,
            abstract_text_field_embedder: TextFieldEmbedder,
            contextualizer: Seq2SeqEncoder,
            num_samples: int = None,
            dropout: float = None) -> None:
        super().__init__(vocab)

        self._abstract_text_field_embedder = abstract_text_field_embedder

        self._seq_embedder = seq_embedder

        # lstm encoder uses PytorchSeq2SeqWrapper for pytorch lstm
        self._contextualizer = contextualizer

        self._forward_dim = contextualizer.get_output_dim()

        # Keep track of the vocab to help BOS/EOS vs. BERT abstract logic
        self._vocab = vocab

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

    def _compute_loss(self,
                      lm_embeddings: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        mask = targets > 0

        non_masked_targets = targets.masked_select(mask) - 1

        non_masked_embeddings = lm_embeddings.masked_select(
                mask.unsqueeze(-1)).view(-1, self._forward_dim)

        return self._softmax_loss(non_masked_embeddings, non_masked_targets)

    def num_layers(self) -> int:
        return self_contextualizer.num_layers + 1

    def forward(self,
                abstracts: Dict[str, torch.LongTensor],
                paper_ids: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the loss from the batch.
        """

        # Embed the abstracts and retrieve <CLS> tokens for each.
        # abstracts (batch, sequence length, #tokens, dims)

        #(batch, seq, dims)
        abstract_embeddings = self._abstract_text_field_embedder(abstracts)[:, :, 0, :]
        
        #(batch, seq, abs_dim + n2v_dim)
        embeddings = torch.cat([abstract_embeddings, self._seq_embedder(paper_ids)], dim=-1)

        # Get text field mask

        #(batch, sequence_length, #tokens)
        mask = get_text_field_mask(abstracts, num_wrapping_dims=1)

        paper_mask = mask.sum(dim=-1) > 0

        contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
                embeddings, paper_mask.long()
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
        })

        return return_dict

    @overrides
    def decode(self,
            output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        perplexity = self._perplexity(output_dict['loss'])
        output_dict['perplexity'] = perplexity
        return output_dict

    def get_metrics(self, reset: bool = False):
                return {"perplexity": self._perplexity.get_metric(reset=reset)}
