from typing import Dict
from overrides import overrides
from collections import defaultdict
import numpy as np
import torch

from allennlp.training.metrics.metric import Metric

@Metric.register("confusion")
class ConfusionMatrix(Metric):

    def __init__(self, num_classes) -> None:
        # matrix shape corresponds to true class, predicted class
        self._matrix = np.zeros((num_classes,num_classes), dtype=int)

    @overrides
    def reset(self):
        self._matrix = np.zeros((num_classes,num_classes), dtype=int)

    @overrides
    def __call__(self, class_probs, labels):
        for probs, label in zip(class_probs.numpy(), labels.numpy()):
            self._matrix[label][np.argmax(probs)] += 1

    @overrides
    def get_metric(self, reset: bool = False):
        ret = np.copy(self._matrix)
        if reset:
            self.reset()
        return ret
