"""
Baselines from ProteinNPT, adapted to better match our setup + use esm2

Note that we wrap ESM in a list so that it doesn't get registered as a
submodule in pytorch. This way the ESM weights won't be saved with the 
checkpoints, as this would slow down training massively.

"""

import torch
import torch.nn as nn
from collections import OrderedDict

from .embedding import LinearFloatEmbedding
from esm.pretrained import load_model_and_alphabet


class CachedESMModel(nn.Module):
    def __init__(self, cache_size=500):
        """
        During eval, the context set will be repeated for each query set, 
        so we use a cache rather than recomputing the ESM embeddings for each query set.
        
        """
        super(CachedESMModel, self).__init__()
        self.cache = OrderedDict()
        self.cache_size = cache_size

    @torch.compiler.disable()
    def forward(self, esm, proteins_tokens):

        esm.to(proteins_tokens.device)
        with torch.no_grad():
            
            # Initialize a list to store embeddings
            embeddings_list = []
            
            # Iterate over each token in the flattened tokens
            for token in proteins_tokens:
                token_tuple = tuple(token.tolist())  # Convert tensor to a hashable type (tuple)
                
                if token_tuple in self.cache:
                    # If token is in cache, use the cached embedding
                    embeddings_list.append(self.cache[token_tuple].to(proteins_tokens.device))
                    self.cache.move_to_end(token_tuple)
                else:
                    # If token is not in cache, process it and store the result in cache
                    embedding = esm(token.unsqueeze(0), repr_layers=[33])['representations'][33].squeeze(0)
                    embeddings_list.append(embedding)
                    self.cache[token_tuple] = embedding.detach().cpu()

                    # Evict the oldest item if the cache exceeds the specified size
                    if len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
            
            # Stack the embeddings to form the final tensor
            proteins_embeddings = torch.stack(embeddings_list)
            
            # Reshape the embeddings to the original shape
            # proteins_embeddings = proteins_embeddings.view(proteins_tokens.shape[0], proteins_tokens.shape[1], proteins_tokens.shape[2], -1)
            
            return proteins_embeddings

class AugmentedPropertyPredictor(nn.Module):

    def __init__(
            self,
            embedding_size: int = 1280,
            hidden_size: int = 1280,
            ff_factor: int = 4,
            num_attention_heads: int = 16,
            padding_idx: int = -1,
            dropout: float = 0.1,
            num_labels: int = 1,
            num_auxiliary_labels: int = 2,
            loss_fn: str ='mse',
            embedding_mode: str = 'esm',
        ):

        super().__init__()

        # self.property_id_embeddings = nn.Embedding(num_labels, embedding_size)

        self.label_pad_id = -110 # NOTE we don't actually have padded labels in this model anyway.
        self.num_labels = num_labels
        self.num_auxiliary_labels = num_auxiliary_labels

        self.embedding_mode = embedding_mode
        if self.embedding_mode == 'esm':
            # self.esm, _ = load_model_and_alphabet('esm2_t33_650M_UR50D')
            self.esm_as_list = [load_model_and_alphabet('esm2_t33_650M_UR50D')[0]] # so that it's not  a submodule.
            self.cached_esm = CachedESMModel(cache_size=1000)
        else:
            raise ValueError(f"Unknown embedding mode {self.embedding_mode}")
        
        self.dropout = nn.Dropout(dropout)

        self.target_head = nn.Linear(embedding_size, 1) # no reason to predict [MASK] token, just 0-1
        self.target_head_auxiliary = nn.Linear(self.num_auxiliary_labels, 1, bias = False)
        torch.nn.init.constant_(self.target_head_auxiliary.weight, 1.0)


        self.loss_fn = nn.functional.mse_loss



    
    def forward(self, 
                proteins_tokens: torch.Tensor,
                sequence_auxiliary_labels: torch.Tensor = None,
                proteins_attention_mask: torch.Tensor = None,
                label_targets: torch.Tensor = None,
                ):
        """

        Args:
            proteins_tokens: (B, L) tokenized protein sequences
            sequence_auxiliary_labels: (B, A) labels
            proteins_attention_mask: (B, L) attention mask for proteins
            label_targets: (B, ) label targets
            return_embeddings: bool, whether to return the embeddings
            use_cache: bool, whether to use a cache for ESM embeddings
        Returns:
            label_logits: (B,) binary label logits
            losses: dict of losses
        """
        
        if proteins_attention_mask is None:
            proteins_attention_mask = torch.zeros_like(proteins_tokens, dtype=bool, device=proteins_tokens.device)

        # # get AA embeddings
        with torch.no_grad():
            # need to flatten B, N for ESM
            proteins_embeddings = self.cached_esm(self.esm_as_list[0], proteins_tokens)
            # proteins_embeddings = self.esm(proteins_tokens, repr_layers=[33])['representations'][33]

        # TODO add 1dconv here?

        x = self.dropout(proteins_embeddings)

        # TODO alternatives to meanpool
        x = x.mean(dim=-2)

        y_hat_from_x = self.target_head(x)
        y_hat_from_0shot = self.target_head_auxiliary(sequence_auxiliary_labels)

        y_hat = y_hat_from_x + y_hat_from_0shot

        y_hat = y_hat.squeeze(-1) # (B,)
        
        loss = None
        if label_targets is not None:
            loss = self.loss_fn(y_hat, label_targets)

        return y_hat, loss



class MLPPropertyPredictor(nn.Module):

    def __init__(
            self,
            embedding_size: int = 1280,
            hidden_size: int = 1280,
            ff_factor: int = 4,
            num_attention_heads: int = 16,
            padding_idx: int = -1,
            dropout: float = 0.1,
            num_labels: int = 1,
            num_auxiliary_labels: int = 2,
            loss_fn: str ='mse',
            embedding_mode: str = 'esm',
        ):

        super().__init__()

        # self.property_id_embeddings = nn.Embedding(num_labels, embedding_size)

        self.label_pad_id = -110 # NOTE we don't actually have padded labels in this model anyway.
        self.num_labels = num_labels

        self.embedding_mode = embedding_mode
        if self.embedding_mode == 'esm':
            # self.esm, _ = load_model_and_alphabet('esm2_t33_650M_UR50D')
            self.esm_as_list = [load_model_and_alphabet('esm2_t33_650M_UR50D')[0]] # so that it's not  a submodule.
            self.cached_esm = CachedESMModel(cache_size=1000)
        else:
            raise ValueError(f"Unknown embedding mode {self.embedding_mode}")
        
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )


        self.loss_fn = nn.functional.mse_loss



    
    def forward(self, 
                proteins_tokens: torch.Tensor,
                sequence_auxiliary_labels: torch.Tensor = None,
                proteins_attention_mask: torch.Tensor = None,
                label_targets: torch.Tensor = None,
                ):
        """

        Args:
            proteins_tokens: (B, L) tokenized protein sequences
            sequence_auxiliary_labels: (B, A) labels
            proteins_attention_mask: (B, L) attention mask for proteins
            label_targets: (B, ) label targets
            return_embeddings: bool, whether to return the embeddings
            use_cache: bool, whether to use a cache for ESM embeddings
        Returns:
            label_logits: (B,) binary label logits
            losses: dict of losses
        """
        
        if proteins_attention_mask is None:
            proteins_attention_mask = torch.zeros_like(proteins_tokens, dtype=bool, device=proteins_tokens.device)

        # # get AA embeddings
        with torch.no_grad():
            # need to flatten B, N for ESM
            proteins_embeddings = self.cached_esm(self.esm_as_list[0], proteins_tokens)
            # proteins_embeddings = self.esm(proteins_tokens, repr_layers=[33])['representations'][33]

        x = self.dropout(proteins_embeddings)
        x = x.mean(dim=-2)

        y_hat = self.mlp(x)

        y_hat = y_hat.squeeze(-1) # (B,)
        
        loss = None
        if label_targets is not None:
            loss = self.loss_fn(y_hat, label_targets)

        return y_hat, loss


