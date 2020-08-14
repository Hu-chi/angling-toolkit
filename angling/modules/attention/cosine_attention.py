import torch
from overrides import overrides

from angling.modules.attention.attention import Attention
from angling.nn.util import tiny_value_of_dtype


@Attention.register("cosine")
class CosineAttention(Attention):
    """
    Computes attention between a vector and a matrix using cosine similarity.

    Registered as an `Attention` with name "cosine".
    """
    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        a_norm = vector / (
                vector.norm(p=2, dim=-1, keepdim=True) + tiny_value_of_dtype(vector.dtype)
        )
        b_norm = matrix / (
                matrix.norm(p=2, dim=-1, keepdim=True) + tiny_value_of_dtype(matrix.dtype)
        )
        return torch.bmm(a_norm.unsqueeze(dim=1), b_norm.transpose(-1, -2)).squeeze(1)
