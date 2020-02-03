from typing import Dict
from copy import deepcopy

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.predictors.predictor import Predictor

@Predictor.register("cocitation")
class CocitationPredictor(Predictor):

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        new_instance = deepcopy(instance)
        outputs = self._model.forward_on_instance(new_instance)
        print(instance)
        print(instance['pid_anchor'])
        output_dict = {
            instance['pid_anchor'].label: outputs['embedding']
        }
        return sanitize(output_dict)
