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
from concite.training.metrics import ConfusionMatrix
from overrides import overrides

@Model.register("acl_edge_classifier")
class AclEdgeClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 node_embedder: TokenEmbedder,
                 null_text_embedder: TokenEmbedder,
                 verbose_metrics: False,
                 classifier_feedforward: FeedForward,
                 use_node_vector: bool = True,
                 use_text: bool = True,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AclEdgeClassifier, self).__init__(vocab, regularizer)

        self.node_embedder = node_embedder
        self.text_field_embedder = text_field_embedder
        # Instead of setting this, omit embedding path in config
        # to get randomly initialized embeddings.
        #self.use_node_vector = use_node_vector
        self.use_text = use_text
        self.null_text_embedder = null_text_embedder
        self.dropout = torch.nn.Dropout(dropout)
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sep_index = self.vocab.get_token_index("[SEP]")

        self.classifier_feedforward = classifier_feedforward

        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}

        self.verbose_metrics = verbose_metrics

        for i in range(self.num_classes):
            self.label_f1_metrics[vocab.get_token_from_index(index=i, namespace="labels")] = F1Measure(positive_label=i)

        self.confusion_matrix = ConfusionMatrix(self.num_classes)

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                citing_paper_id: torch.LongTensor,
                cited_paper_id: torch.LongTensor,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        if self.use_text:
            mask = text == self.sep_index
            embedded_text = self.text_field_embedder(text)[:, self.sep_index, :]
        else:
            # Initialize random per-paper_id vectors to stand in for BERT
            # vectors
            embedded_text = self.null_text_embedder(citing_paper_id)
            embedded_text = self.null_text_embedder(cited_paper_id)
        node_vector = torch.cat([self.node_embedder(cited_paper_id), self.node_embedder(citing_paper_id)], dim=-1)
        logits = self.classifier_feedforward(self.dropout(torch.cat([embedded_text, node_vector], dim=-1)))
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
        class_probs = F.softmax(output_dict['logits'], dim=-1)
        self.confusion_matrix(class_probs, output_dict['labels'])
        output_dict['confusion_matrix'] = self.confusion_matrix.get_metric()
        output_dict['class_probs'] = class_probs
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
