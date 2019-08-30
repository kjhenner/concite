import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("lookup_embedder")
class LookupTokenEmbedder(TokenEmbedder):
    """
    Loads a lookup dictionary of embeddings from a file
    ----------
    hidden_dim : `int`, required.
    file_path : `str`, required.
    """
    def __init__(self,
            vocab: Vocabulary,
            vocab_namespace: str,
            hidden_dim: int) -> None:
        self.hidden_dim = hidden_dim
        super().__init__()

    def get_output_dim(self):
        return self.hidden_dim

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> torch.Tensor:
        return inputs
