import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from .esm.rotary_embedding import RotaryEmbedding

from .esm.modules import (
    ESM1bLayerNorm,
    ESM1LayerNorm,
    gelu,
)


class SetMultiheadAttention(nn.Module):
    """Multi-headed attention.

    This runs on (L, B, N, H) tensors, where L is the sequence length, B is the batch size,
    N is the set size, and H is the hidden dimension.

    Adapted from ESM, but got rid of most functionality. Always uses torch attention.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        encoder_decoder_attention: bool = False,
        use_rotary_embeddings: bool = False,
        all_seq_attn: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.encoder_decoder_attention = encoder_decoder_attention


        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn


        self.reset_parameters()

        self.onnx_trace = False
        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

        self.all_seq_attn = all_seq_attn

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        # incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        # need_weights: bool = True,
        # static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """

        bsz, set_sz, tgt_len, embed_dim = query.size()
        _, _, src_len, _ = key.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [bsz, set_sz, tgt_len, embed_dim]


        if self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, set_sz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, set_sz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )


        # reshape to get heads
        q = q.contiguous().view(bsz,set_sz, tgt_len, self.num_heads, self.head_dim).transpose(2, 3) # bsz, set_sz, num_heads, tgt_len, head_dim
        k = k.contiguous().view(bsz, set_sz, -1, self.num_heads, self.head_dim).transpose(2, 3) # bsz, set_sz, num_heads, tgt_len, head_dim
        v = v.contiguous().view(bsz, set_sz, -1, self.num_heads, self.head_dim).transpose(2, 3) # bsz, set_sz, num_heads, tgt_len, head_dim

        if self.rot_emb:
            # taken straight from esm, make sure we call with right tensor shape:
            # original q reshape call: q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            q, k = self.rot_emb(
                q.contiguous().view(bsz * set_sz * self.num_heads, tgt_len, self.head_dim),
                k.contiguous().view(bsz * set_sz * self.num_heads, -1, self.head_dim),
                )
            q = q.view(bsz, set_sz, self.num_heads, tgt_len, self.head_dim)
            k = k.view(bsz, set_sz, self.num_heads, -1, self.head_dim)

        if self.all_seq_attn:
            # flatten the keys and values so that they are all the same set_sz*src_len for all sets
            #k, v : (bsz, set_sz, num_heads, src_len, head_dim) -> (bsz, num_heads, set_sz*src_len, head_dim)
            k = k.transpose(1, 2).contiguous().view(bsz, self.num_heads, set_sz*src_len, self.head_dim) # bsz, num_heads, set_sz*src_len, head_dim
            v = v.transpose(1, 2).contiguous().view(bsz, self.num_heads, set_sz*src_len, self.head_dim) # bsz, num_heads, set_sz*src_len, head_dim

            # repeat k,v so that all queries can attend to the same keys
            k = k.unsqueeze(1).expand(-1, set_sz, -1, -1, -1)
            v = v.unsqueeze(1).expand(-1, set_sz, -1, -1, -1)

            # no need for mask here, all keys are valid
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)

        else:
            # Expand the key padding mask to match the dimensions required by the attention function
            if key_padding_mask is not None:
                key_padding_mask_ext = key_padding_mask.view(bsz, set_sz, 1, 1, tgt_len).expand(-1, -1, self.num_heads, tgt_len, -1)  # (bsz, set_sz, num_heads, tgt_len, tgt_len)
                key_padding_mask_ext = ~key_padding_mask_ext

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=key_padding_mask_ext, dropout_p=0.0, is_causal=False, scale=None) # bsize, set_sz, num_heads, tgt_len, head_dim
        
        attn_output = attn_output.transpose(2, 3).contiguous().view(bsz, set_sz, tgt_len, embed_dim)
        return self.out_proj(attn_output), None

    
class AttentionResamplingSequencePooler(nn.Module):
    """
    Performs a mean pool and uses that as queries to get an attention-sampled pooled representation.x
    """
    def __init__(
            self,
            embed_dim,
            num_embeddings,
            ):
        
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim * num_embeddings)
        self.num_embeddings = num_embeddings

    def forward(self, x, key_padding_mask = None):
        """
        Input: x (bsize, set_sz, seq_len, embed_dim)
        Output: x_pooled (bsize, set_sz, num_embeddings, embed_dim)
        """
        # bsz, set_sz, tgt_len, embed_dim = x.size()
        # mean pool
        x_mean = x.mean(dim=2)
        # project mean pool to queries
        q = self.q_proj(x_mean)
        q = q.view(q.size(0), q.size(1), self.num_embeddings, -1) # bsize, set_sz, num_embeddings, embed_dim

        if key_padding_mask is not None:
                # expand to shape (bsize, set_sze, num_embeddings, tgt_len)
                key_padding_mask_ext = key_padding_mask.unsqueeze(2).expand(-1, -1, self.num_embeddings, -1)  # (bsz, set_sz, num_embeddings, tgt_len)
                key_padding_mask_ext = ~key_padding_mask_ext  # Invert the mask, torch wants True where unpadded

        x_pooled = F.scaled_dot_product_attention(q, x, x, attn_mask=key_padding_mask_ext, dropout_p=0.0, is_causal=False, scale=None)
        return x_pooled

class MeanSequencePooler(nn.Module):
    """
    Performs a mean pool over the sequence dimension.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, key_padding_mask = None):
        """
        Input: x (bsize, set_sz, seq_len, embed_dim)
        Output: x_pooled (bsize, set_sz, 1, embed_dim)
        """
        x_pooled = x.mean(dim=2, keepdim=True)
        return x_pooled

class PooledAttentionLayer(nn.Module):
    """
    Per-sequence attention, followed by all-sequence attention. 
    PoET approach, using modules taken from ESM.
    """
    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        use_rotary_embeddings: bool = False,
        dropout: float = 0,
        pooling_method: str = "attention",
        pooling_kwargs: Optional[Dict] = {'num_embeddings': 3},
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self.dropout = nn.Dropout(dropout)

        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn_seq = SetMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        if pooling_method == "attention":
            self.pooler = AttentionResamplingSequencePooler(
                embed_dim=self.embed_dim,
                **pooling_kwargs,
            )
        elif pooling_method == "mean":
            self.pooler = MeanSequencePooler()
        else:
            raise ValueError(f"Unknown pooling method {pooling_method}")

        self.self_attn_set = SetMultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
            all_seq_attn=True, # performs flattening of key and value sets
        )
        self.self_attn_layer_norm_1 = BertLayerNorm(self.embed_dim)

        self.self_attn_layer_norm_2 = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self, x, self_attn_mask=None, self_attn_padding_mask=None
    ):
        B, N, L, H = x.size()
        # sequence-independent attention
        # x = x.view(B * N, L, H)

        residual = x
        # NOTE layernorm operates on all trailing dimensions - need to reshape set dim into batch dim
        x = self.self_attn_layer_norm_1(x.view(B * N, L, H)).view(B, N, L, H)
        x, attn = self.self_attn_seq(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
        )
        x = residual + self.dropout(x)

        # attention over all sequences
        x_pooled = self.pooler(x, self_attn_padding_mask)
        # x = x.view(B, N * L, H)
        residual = x
        x = self.self_attn_layer_norm_2(x.view(B * N, L, H)).view(B, N, L, H)
        x, attn = self.self_attn_set(
            query=x,
            key=x_pooled,#.unsqueeze(1)#.expand(-1, N, -1, -1).contiguous(),
            value=x_pooled,#.unsqueeze(1)#.expand(-1, N, -1, -1).contiguous(),
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
        )
        x = residual + self.dropout(x)

        # reshape x back
        # x = x.view(B, N, L, H)

        residual = x
        x = self.final_layer_norm(x.view(B * N, L, H)).view(B, N, L, H)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + self.dropout(x)

        return x, attn