
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
from allennlp.modules import FeedForward, TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides

@Model.register("venue_classifier")
class VenueClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 verbose_metrics: False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(VenueClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward

        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}

        self.verbose_metrics = verbose_metrics

        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)

        self.loss = torch.nn.CrossEntropyLoss()

        self.pool = lambda text, mask: util.get_final_encoder_states(text, mask, bidirectional=True)

        initializer(self)

    @overrides
    def forward(self,
                abstract: Dict[str, torch.LongTensor],
                graph_vector: torch.LongTensor,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        embedded_abstract = self.text_field_embedder(abstract)
        mask = util.get_text_field_mask(abstract)
        encoded_abstract = self.text_encoder(embedded_abstract, mask)
        pooled = self.pool(encoded_abstract, mask)

        logits = self.classifier_feedforward(torch.cat([pooled, graph_vector], dim=-1))
        class_probs = F.softmax(logits, dim=1)

        output_dict = {"logits": logits}

        if label is not None:
            loss = self.loss(logits, label)
            output_dict["loss"] = loss

            for i in range(self.num_classes):
                metric = self.label_f1_metrics[self.vocab.get_token_from_index(index=i, namespace="labels")]
                metric(class_probs, label)
            self.label_accuracy(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probs'] = class_probabilities
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}

        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            if self.verbose_metrics:
                metric_dict[name + '_P'] = metric_val[0]
                metric_dict[name + '_R'] = metric_val[1]
                metric_dict[name + '_F1'] = metric_val[2]
            sum_f1 += metric_val[2]

        names = list(self.label_f1_metrics.keys())
        total_len = len(names)
        average_f1 = sum_f1 / total_len
        metric_dict['average_F1'] = average_f1
        metric_dict['accuracy'] = self.label_accuracy.get_metric(reset)
        return metric_dict
