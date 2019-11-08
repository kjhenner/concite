from typing import Dict
from copy import deepcopy

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.predictors.predictor import Predictor

@Predictor.register("acl_classifier")
class AclClassifierPredictor(Predictor):

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        new_instance = deepcopy(instance)
        outputs = self._model.forward_on_instance(new_instance)
        return sanitize(outputs)
