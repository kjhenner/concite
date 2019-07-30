
from typing import Dict, Optional

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

@Model.register("venue_classifier")
class VenueClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 classifier_feedforward: FeedForward,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VenueClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.dropout = torch.nn.Dropout(dropout)
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.classifier_feedforward = classifier_feedforward

        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}

        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=1, namespace="labels")] = F1Measure(positive_label=1)

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,
                abstract: Dict[str, torch.LongTensor],
                graph_vector: torch.LongTensor,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        embedded_abstract = self.text_field_embedder(abstract)
        pooled = self.dropout(embedded_abstract[:, 0, :])
        logits = self.classifier_feedforward(torch.cat([pooled, graph_vector], dim=-1))

        class_probs = F.softmax(logits, dim=1)

        output_dict = {"logits": logits}

        if label is not None:
            loss = self.loss(logits, label)
            output_dict["loss"] = loss

            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=1, namespace="labels")]
                metric(class_probs, label)
            self.label_accuracy(logits, label)

        return output_dict

#    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#        predictions = output_dict['class_probabilities'].cpu().data.numpy()
#        argmax_indices = numpy.argmax(predictions, axis=-1)
#        labels = [self.vocab.get_token_from_index(x, namespace="labels")
#                for x in argmax_indices]
#        output_dict['label'] = labels
#        return output_dict
