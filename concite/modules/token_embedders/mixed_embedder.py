from overrides import overrides
import numpy
import torch
import logging
import warnings
import itertools
import jsonlines

from torch.nn.functional import embedding

from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.nn import util

logger = logging.getLogger(__name__)

@TokenEmbedder.register("mixed_embedder")
class MixedEmbedder(Embedding):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 projection_dim: int = None,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 vocab_namespace: str = None,
                 pretrained_file: str = None) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = vocab_namespace
        self._pretrained_file = pretrained_file 

        self.output_dim = projection_dim or embedding_dim

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise ConfigurationError("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None

    @overrides
    def forward(self, inputs):
        original_size = inputs.size()
        inputs = util.combine_initial_dims(inputs)

        embedded = embedding(inputs, self.weight,
                             padding_idx=self.padding_index,
                             max_norm=self.max_norm,
                             norm_type=self.norm_type,
                             scale_grad_by_freq=self.scale_grad_by_freq,
                             sparse=self.sparse)

        embedded = util.uncombine_initial_dims(embedded, original_size)

        return embedded

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'MixedEmbedder':
        num_embeddings = params.pop_int('num_embeddings', None)
        vocab_namespace = params.pop("vocab_namespace", None if num_embeddings else "tokens")
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(vocab_namespace)
        embedding_dim = params.pop_int('embedding_dim')
        pretrained_file = params.pop('pretrained_file', None)
        projection_dim = params.pop_int('projection_dim', None)
        trainable = params.pop_bool('trainable', True)
        padding_index = params.pop_int('padding_index', None)
        max_norm = params.pop_float('max_norm', None)
        norm_type = params.pop_float('norm_type', 2.)
        scale_grad_by_freq = params.pop_bool('scale_grad_by_freq', False)
        sparse = params.pop_bool('sparse', False)
        params.assert_empty(cls.__name__)

        if pretrained_file:
            weight = _read_embeddings_from_jsonl(pretrained_file,
                                                 embedding_dim,
                                                 vocab,
                                                 vocab_namespace)
        else:
            weight = None

        return cls(num_embeddings=num_embeddings,
                   embedding_dim=embedding_dim,
                   projection_dim=projection_dim,
                   weight=weight,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse,
                   vocab_namespace=vocab_namespace)

def _read_embeddings_from_jsonl(embeddings_filename: str,
                                embedding_dim: int,
                                vocab: Vocabulary,
                                namespace: str = 'tokens') -> torch.FloatTensor: 

    tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)

    embeddings = {}

    with jsonlines.open(embeddings_filename) as reader:
        for instance in reader:
            token = instance['paper_id']
            graph_vector = numpy.asarray(instance['graph_vector'])
            vector = numpy.asarray(instance['graph_vector'], dtype='float32')
            if len(vector) != embedding_dim:
                logger.warning("Found instance with wrong number of dimensions (expected: %d; actual %d): %s", embedding_dim, len(vector), token)
            else:
                embeddings[token] = vector
    all_embeddings = numpy.asarray(list(embeddings.values()))
    embeddings_mean = float(numpy.mean(all_embeddings))
    embeddings_std = float(numpy.std(all_embeddings))
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_std)
    num_tokens_found = 0
    for i in range(vocab_size):
        token = vocab.get_index_to_token_vocabulary(namespace)[i]
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
            num_tokens_found += 1
        else:
            logger.debug("Token %s was not found in the embedding file. Initializing randomly.", token)

    logger.info("pretrained embeddings were found for %d out of %d tokens", num_tokens_found, vocab_size)

    return embedding_matrix
