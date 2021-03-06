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

@Model.register("acl_classifier")
class AclClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 node_embedder: TokenEmbedder,
                 verbose_metrics: False,
                 classifier_feedforward: FeedForward,
                 use_node_vector: bool = True,
                 use_abstract: bool = True,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AclClassifier, self).__init__(vocab, regularizer)

        self.node_embedder = node_embedder
        self.text_field_embedder = text_field_embedder
        self.use_node_vector = use_node_vector
        self.use_abstract = use_abstract
        self.dropout = torch.nn.Dropout(dropout)
        self.num_classes = self.vocab.get_vocab_size("labels")

        self.classifier_feedforward = classifier_feedforward

        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}

        self.verbose_metrics = verbose_metrics

        for i in range(self.num_classes):
            label_name = vocab.get_token_from_index(index=i, namespace="labels")
            self.label_f1_metrics[label_name] = F1Measure(positive_label=i)

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,
                abstract: Dict[str, torch.LongTensor],
                paper_id: torch.LongTensor,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        if self.use_abstract and self.use_node_vector:
            embedding = torch.cat([
                self.text_field_embedder(abstract)[:, 0, :],
                self.node_embedder(paper_id)], dim=-1)
        elif self.use_abstract:
            embedding = self.text_field_embedder(abstract)[:, 0, :]
        elif self.use_node_vector:
            embedding = self.node_embedder(paper_id)
        else:
            embedding = self.node_embedder(paper_id)

        logits = self.classifier_feedforward(self.dropout(embedding))
        class_probs = F.softmax(logits, dim=1)
        output_dict = {"logits": logits}

        if label is not None:
            loss = self.loss(logits, label)
            output_dict["label"] = label
            output_dict["loss"] = loss
        for i in range(self.num_classes):
            label_name = self.vocab.get_token_from_index(index=i, namespace="labels")
            metric = self.label_f1_metrics[label_name]
            metric(class_probs, label)
        self.label_accuracy(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probs = F.softmax(output_dict['logits'], dim=-1)
        output_dict['pred_label'] = [
            self.vocab.get_token_from_index(index=int(np.argmax(probs)), namespace="labels")
            for probs in class_probs.cpu()
        ]
        output_dict['label'] = [
            self.vocab.get_token_from_index(index=int(label), namespace="labels")
            for label in output_dict['label'].cpu()
        ]
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
        if total_len > 0:
            average_f1 = sum_f1 / total_len
        else:
            average_f1 = 0.0
        metric_dict['average_F1'] = average_f1
        metric_dict['accuracy'] = self.label_accuracy.get_metric(reset)
        return metric_dict
