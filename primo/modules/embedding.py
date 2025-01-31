import torch
from torch import nn
from typing import Optional

class LinearFloatEmbedding(nn.Module):
    """
    Embedding layer for counts or normalized counts.
    Uses a linear layer to map floats to embeddings.

    This keeps a separate linear layer for each dimension
    in ``num_embeddings``.

    Parameters
    ----------
    embedding_dim : int
        Dimension of embeddings.
    num_embeddings : int
        Number of embeddings.
    padding_idx : int, optional
        Int representing the padding token. Default is -100.
    mask_idx : int, optional
        Int representing the mask token. Default is -1.

    Attributes
    ----------
    mask_token_weight : nn.Parameter
        Parameter representing the mask token weight.
    float_to_embedding : nn.Linear
        Linear layer for converting floats to embeddings.
    """
    def __init__(
        self, 
        embedding_dim: int,
        num_embeddings: int,
        padding_idx: Optional[int] = None, 
        mask_idx: Optional[int] = None,
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx

        self.float_to_embedding = nn.ModuleList(
            [nn.Linear(1, embedding_dim) for _ in range(num_embeddings)]
        )

        # NOTE tried doing this with nn.Parameters, but had precision issues.
        if self.padding_idx is not None:
            self.padding_token_weight = nn.Embedding(1, embedding_dim)
            self.padding_token_weight.weight.data.fill_(0.)  # Fill with zeros
        if self.mask_idx is not None:
            self.mask_token_weight = nn.Embedding(1, embedding_dim) 


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        assert input.shape[-1] == self.num_embeddings, f"Input shape {input.shape} with {input.shape[-1]} does not match num_embeddings {self.num_embeddings}"

        # emb = torch.stack([self.float_to_embedding[i](input[:, i].unsqueeze(-1)) for i in range(self.num_embeddings)], dim=1)
        # always use last dim.
        emb = torch.stack([self.float_to_embedding[i](input[..., i].unsqueeze(-1)) for i in range(self.num_embeddings)], dim=-2)

        # replace emb with mask_token_weight where input == mask_idx
        if self.padding_idx is not None:
            # emb[input == self.mask_idx] = self.paddding_token_weight.to(emb.dtype)
            emb[input == self.padding_idx] = self.padding_token_weight(torch.zeros(1).long().to(input.device)).to(emb.dtype)

        if self.mask_idx is not None:
            # emb[input == self.mask_idx] = self.mask_token_weight.to(emb.dtype)
            emb[input == self.mask_idx] = self.mask_token_weight(torch.zeros(1).long().to(input.device)).to(emb.dtype)

        return emb