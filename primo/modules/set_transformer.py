import torch
import torch.nn as nn
from collections import OrderedDict


from .esm.modules import RobertaLMHead
from .tiered_attention import TieredAttentionLayer
from .pooled_cross_attention import PooledAttentionLayer
from .pooled_summary_attention import PooledSummaryVectorAttentionLayer
from .embedding import LinearFloatEmbedding
from esm.pretrained import load_model_and_alphabet

def squashed_mse_loss(input, target, min_val=-0.5, max_val=0.5, reduction='mean'):
    """
    Compute the Mean Squared Error (MSE) loss with squashing of values outside the specified range.

    Parameters:
    - input (Tensor): The input tensor.
    - target (Tensor): The target tensor.
    - min_val (float): The minimum value for squashing. Default is -0.5.
    - max_val (float): The maximum value for squashing. Default is 0.5.
    - reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.

    Returns:
    - Tensor: The computed MSE loss.
    """
    # Squash the input and target values to the specified range
    input_squashed = torch.clamp(input, min_val, max_val)
    target_squashed = torch.clamp(target, min_val, max_val)
    
    # Compute the MSE loss
    loss = torch.nn.functional.mse_loss(input_squashed, target_squashed, reduction=reduction)
    
    return loss


def all_pairs_loss(input, target, mask=None, reduction='mean', mask_logic='or'):
    """
    Compute a pairwise loss for all pairs in the data.

    Args:
        input (torch.Tensor): The input tensor of shape (batch_size, num_elements).
        target (torch.Tensor): The target tensor of shape (batch_size, num_elements).
        mask (torch.Tensor): The mask tensor of shape (batch_size, num_elements) indicating which elements to ignore. True means ignore,
                                False means keep. If None, no mask will be applied.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                         'none': no reduction will be applied,
                         'mean': the sum of the output will be divided by the number of elements in the output,
                         'sum': the output will be summed.
        mask_logic (str): Specifies the logic to apply to the mask: 'or' | 'and'. If or, a pair that has at least one masked element will be ignored.
                            If and, a pair that has both masked elements will be ignored.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Compute all log(sigmoid(input_i - input_j)) for all i, j
    input_diff = input.unsqueeze(2) - input.unsqueeze(1)  # Shape: (batch_size, num_elements, num_elements)
    log_sigmoid_diff = torch.nn.functional.logsigmoid(input_diff)  # Shape: (batch_size, num_elements, num_elements)

    # Compute all 1 if target_i > target_j, 0 otherwise
    target_diff = target.unsqueeze(2) - target.unsqueeze(1)  # Shape: (batch_size, num_elements, num_elements)
    indicator = (target_diff > 0).float()  # Shape: (batch_size, num_elements, num_elements)

    # Compute the loss: L = - Indicator(target_i > target_j) * log(sigmoid(input_i - input_j))
    loss_matrix = -indicator * log_sigmoid_diff  # Shape: (batch_size, num_elements, num_elements)

    # make a mask and apply it - if either i or j is masked, set loss to 0.
    if mask is not None:
        mask_i = mask.unsqueeze(2)  # Shape: (batch_size, num_elements, 1)
        mask_j = mask.unsqueeze(1)  # Shape: (batch_size, 1, num_elements)

        if mask_logic == 'or':
            mask_pairs = mask_i | mask_j  # Shape: (batch_size, num_elements, num_elements)
        elif mask_logic == 'and':
            mask_pairs = mask_i & mask_j  # Shape: (batch_size, num_elements, num_elements)
        else:
            raise ValueError("mask_logic must be 'or' or 'and'")
        
        loss_matrix = loss_matrix * (~mask_pairs).float()  # Apply mask
    # Sum over i, j for each element of batch so that we get (batch_size,) loss
    loss = loss_matrix.sum(dim=(1, 2))  # Shape: (batch_size,)

    # Apply reduction
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss



class CachedESMModel(nn.Module):
    def __init__(self, esm_model, cache_size=500):
        """
        During eval, the context set will be repeated for each query set, 
        so we use a cache rather than recomputing the ESM embeddings for each query set.
        
        """
        super(CachedESMModel, self).__init__()
        self.cache = OrderedDict()
        self.cache_size = cache_size

    @torch.compiler.disable()
    def forward(self, esm, proteins_tokens):
        with torch.no_grad():
            # Flatten B, N for ESM
            proteins_tokens_flat = proteins_tokens.view(-1, proteins_tokens.shape[-1])
            
            # Initialize a list to store embeddings
            embeddings_list = []
            
            # Iterate over each token in the flattened tokens
            for token in proteins_tokens_flat:
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
            proteins_embeddings = proteins_embeddings.view(proteins_tokens.shape[0], proteins_tokens.shape[1], proteins_tokens.shape[2], -1)
            
            return proteins_embeddings



class ProteinSetTransformer(nn.Module):

    def __init__(
            self,
            num_layers: int,
            embedding_size: int = 1280,
            hidden_size: int = 1280,
            ff_factor: int = 4,
            num_attention_heads: int = 16,
            padding_idx: int = -1,
            dropout: float = 0.1,
            num_labels: int = 1,
            num_auxiliary_labels: int = 2,
            target_predict_mode='embedding',
            loss_fn='mse',
            attn_method = 'tiered',
            auxiliary_label_mode: str = 'embed',
        ):

        super().__init__()

        self.property_id_embeddings = nn.Embedding(num_labels, embedding_size)

        self.label_pad_id = -110 # NOTE we don't actually have padded labels in this model anyway.
        self.label_embeddings = LinearFloatEmbedding(embedding_size, 1, padding_idx=self.label_pad_id, mask_idx=-100)
        self.auxiliary_label_mode = auxiliary_label_mode
        if auxiliary_label_mode == 'embed':
            self.auxiliary_label_embeddings = LinearFloatEmbedding(embedding_size, num_auxiliary_labels, padding_idx=self.label_pad_id, mask_idx=-100)
        elif auxiliary_label_mode == 'add':
            # prepare parameter at default initialization value 0.5
            raise NotImplementedError('auxiliary_label_mode `add` not done yet')
            self.auxiliary_mixing_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        elif auxiliary_label_mode == 'embed_and_add':
            self.auxiliary_label_embeddings = LinearFloatEmbedding(embedding_size, num_auxiliary_labels, padding_idx=self.label_pad_id, mask_idx=-100)
            self.auxiliary_mixing_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
            
        else:
            raise ValueError(f'auxiliary_label_mode {auxiliary_label_mode} not implemented')
        self.num_labels = num_labels
        self.num_auxiliary_labels = num_auxiliary_labels


        self.esm, _ = load_model_and_alphabet('esm2_t33_650M_UR50D')
        self.cached_esm = CachedESMModel(self.esm)

        if embedding_size != hidden_size:
            self.in_projection = nn.Linear(embedding_size, hidden_size)
            self.out_projection = nn.Linear(hidden_size, embedding_size)
        else:
            self.in_projection = None
            self.out_projection = None

        if attn_method == 'tiered':
            attn_layer = TieredAttentionLayer
        elif attn_method == 'pooled':
            attn_layer = PooledAttentionLayer
        elif attn_method == 'pooled_summary':
            attn_layer = PooledSummaryVectorAttentionLayer
            self.num_summary_vectors = 1
        else:
            raise NotImplementedError()
        self.attn_method = attn_method

        self.transformer = nn.ModuleList([
            attn_layer(
                embed_dim=hidden_size,
                attention_heads=num_attention_heads,
                ffn_embed_dim=hidden_size*ff_factor,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.esm_embedding_weights = nn.Parameter(self.esm.embed_tokens.weight.data.clone(), requires_grad=False)
        self.lm_head = RobertaLMHead(
            embed_dim=embedding_size,
            output_dim=self.esm_embedding_weights.shape[0],
            weight = self.esm_embedding_weights
        )

        self.target_predict_mode = target_predict_mode
        self.target_head = nn.Linear(embedding_size, 1) # no reason to predict [MASK] token, just 0-1
        # assert target_predict_mode in ['embedding', 'mean_concat'], "target_predict_mode must be 'embedding' or 'mean_concat'"
        # self.target_predict_mode = target_predict_mode
        # if self.target_predict_mode == 'embedding':
        #     self.target_head = RobertaLMHead(hidden_size, 1)
        # elif self.target_predict_mode == 'mean_concat':
        #     self.target_head = RobertaLMHead(hidden_size*2, self.num_labels) # 2 mean pools
        if loss_fn == 'mse':
            self.loss_fn = nn.functional.mse_loss
        elif loss_fn == 'squashed_mse_0.5':
            self.loss_fn = squashed_mse_loss
        elif loss_fn == 'all_pairs':
            self.loss_fn = all_pairs_loss
        else:
            raise NotImplementedError()




    
    def forward(self, 
                proteins_tokens: torch.Tensor,
                property_ids: torch.Tensor,
                sequence_labels: torch.Tensor,
                sequence_auxiliary_labels: torch.Tensor = None,
                proteins_attention_mask: torch.Tensor = None,
                proteins_targets: torch.Tensor = None,
                label_targets: torch.Tensor = None,
                return_embeddings: bool = False,
                use_cache: bool = False
                ):
        """
        Operates on batches of sets of proteins.

        Args:
            proteins_tokens: (B, N, L) tokenized protein sequences
            property_ids: (B, N) property id for each protein sequence
            sequence_labels: (B, N) label for each protein sequence
            sequence_auxiliary_labels: (B, N, A) binary labels
            proteins_attention_mask: (B, N, L) attention mask for proteins
            proteins_targets: (B, N, L) masked language model targets for proteins
            label_targets: (B, N) label targets
            return_embeddings: bool, whether to return the embeddings
            use_cache: bool, whether to use a cache for ESM embeddings
        Returns:
            protein_logits: (B, L, T, 33) masked language model logits for proteins
            label_logits: (B, N) binary label logits
            losses: dict of losses
        """
        
        if proteins_attention_mask is None:
            proteins_attention_mask = torch.zeros_like(proteins_tokens, dtype=bool, device=proteins_tokens.device)

        # # get AA embeddings
        with torch.no_grad():
            # need to flatten B, N for ESM
            proteins_tokens_flat = proteins_tokens.view(-1, proteins_tokens.shape[-1])
            if use_cache:
                proteins_embeddings = self.cached_esm(self.esm, proteins_tokens)
            else:
                proteins_embeddings = self.esm(proteins_tokens_flat, repr_layers=[33])['representations'][33]
            proteins_embeddings = proteins_embeddings.view(proteins_tokens.shape[0], proteins_tokens.shape[1], proteins_tokens.shape[2], -1)
        property_type_embeddings = self.property_id_embeddings(property_ids).unsqueeze(2) # (B, N, 1, H)
        sequence_label_embeddings = self.label_embeddings(sequence_labels.unsqueeze(-1)) # (B, N, 1, H)
        auxiliary_label_embeddings = self.auxiliary_label_embeddings(sequence_auxiliary_labels) # (B, N, A, H)

        input_embeddings = torch.cat([property_type_embeddings, sequence_label_embeddings, auxiliary_label_embeddings, proteins_embeddings], dim=2)
        attention_mask = torch.cat([torch.zeros_like(property_type_embeddings[:,:,:,0], dtype=bool), torch.zeros_like(sequence_label_embeddings[:,:,:,0], dtype=bool), torch.zeros_like(auxiliary_label_embeddings[:,:,:,0], dtype=bool), proteins_attention_mask], dim=2)


        # run transformer
        # ESM operates in seq_len x batch_size x hidden_size
        # but masks still need to be batch_size x seq_len
        # TODO double check whether this is accurate with the custom transformer stack

        if self.in_projection is not None:
            input_embeddings = self.in_projection(input_embeddings)

        if self.attn_method == 'pooled_summary':
            # add pool vectors to each protein sequence
            pooled_summary_vectors = torch.mean(input_embeddings, dim=2, keepdim=True) # (B, N, 1, H)
            pooled_summary_vectors = pooled_summary_vectors.repeat(1, 1, self.num_summary_vectors, 1)  # (B, N, P, H)
            input_embeddings = torch.cat([pooled_summary_vectors, input_embeddings], dim=2) # (B, N, L+P, H)

            attention_mask = torch.cat([torch.zeros_like(pooled_summary_vectors[:,:,:,0], dtype=bool), attention_mask], dim=2)

        output = input_embeddings
        for layer in self.transformer:
            output, attn = layer(output, self_attn_padding_mask=attention_mask)

        if self.attn_method == 'pooled_summary':
            output = output[:, :, self.num_summary_vectors:, :] # (B, N, L, H)

        if self.out_projection is not None:
            output = self.out_projection(output)

        # output is same shape as input_embeddings (B, N, L, H)

        # separate outputs
        start_pos = 0
        poperty_type_output = output[:,:, start_pos:property_type_embeddings.shape[2]]
        start_pos += property_type_embeddings.shape[2]
        label_output = output[:, :, start_pos:start_pos+sequence_label_embeddings.shape[2]]
        start_pos += sequence_label_embeddings.shape[2]
        auxiliary_label_output = output[:, :, start_pos:start_pos+auxiliary_label_embeddings.shape[2]]
        start_pos += auxiliary_label_embeddings.shape[2]
        proteins_output = output[:,:, start_pos:]

        # MLM head
        proteins_logits = self.lm_head(proteins_output) # (B, N, L, 33)

        # label prediction
        if self.target_predict_mode == 'embedding':
            label_preds = self.target_head(label_output) # (B, N, 1)
        elif self.target_predict_mode == 'mean_concat':
            raise NotImplementedError()
        
        # mix in auxiliary labels if we have to
        # NOTE we have a mean here to generalize to N>1 auxiliary label usage. Untested.
        if self.auxiliary_label_mode in ['add', 'embed_and_add']:
            # label_preds = (label_preds * self.auxiliary_mixing_alpha) + (sequence_auxiliary_labels.mean(dim=-1, keepdim=True).unsqueeze(-1) * (1-self.auxiliary_mixing_alpha))
            label_preds = (label_preds * self.auxiliary_mixing_alpha) + sequence_auxiliary_labels.mean(dim=-1, keepdim=True).unsqueeze(-1) 


        # losses
        losses = {}
        if proteins_targets is not None:
            protein_loss = torch.nn.functional.cross_entropy(
                proteins_logits.reshape(-1, proteins_logits.shape[-1]),
                proteins_targets.reshape(-1),
                ignore_index=-1
                )
            losses["protein"] = protein_loss

        if label_targets is not None:
            if self.loss_fn == all_pairs_loss:
                label_loss = self.loss_fn(
                    label_preds.squeeze((-2,-1)), # trailing seq dim and hidden dim
                    label_targets,
                    # only take loss for samples that actually had a masked label
                    mask= sequence_labels != self.label_embeddings.mask_idx
                    )
            else:
                preds_flat = label_preds.reshape(-1)
                targets_flat = label_targets.reshape(-1)
                sequence_labels_flat = sequence_labels.reshape(-1)
                preds_flat, targets_flat = preds_flat[sequence_labels_flat != self.label_pad_id], targets_flat[sequence_labels_flat != self.label_pad_id]
                label_loss = self.loss_fn(
                    # only take loss for samples that actually had a masked label
                    preds_flat,
                    targets_flat,
                    reduction='none'
                    )
                label_loss = label_loss[sequence_labels_flat == self.label_embeddings.mask_idx].mean()
            losses["label"] = label_loss

        label_preds = label_preds.squeeze((-1,-2)) # discard trailing seq dim and hidden dim
        if return_embeddings:
            return proteins_logits, label_preds, losses, proteins_output, label_output
        else:
            return proteins_logits, label_preds, losses


