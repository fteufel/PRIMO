"""
Classes for masking (embedding_1, embedding_2, labels) tuples.

Implement different masking strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Iterable
import torch



class BaseMasker(ABC):

    def __init__(self, label_mask_token_id: int, aa_mask_token_id: int) -> None:
        self.label_mask_token_id = label_mask_token_id
        self.aa_mask_token_id = aa_mask_token_id

    @abstractmethod
    def mask(self, sequence_tokens: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask amino acids and labels in the embeddings and labels.

        Args:
            sequence_tokens (torch.Tensor): tokenized amino acid sequence for protein - this includes the CLS and SEP tokens
            label (torch.Tensor): label for the protein.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked sequence and label
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mask(*args, **kwargs)


# class RandomMasker(BaseMasker):
    # def __init__(self,
    #              label_mask_token_id: int =2,
    #              aa_mask_token_id: int = 32,
    #              aa_mask_prob: float = 0.15,
    #              label_mask_prob: float = 0.15,
    #              mask_both_proteins: bool = False,
    #              always_mask_variants: bool = False
    #              ) -> None:
    #     """
    #     Mask amino acids and labels randomly with a given probability.

    #     Args:
    #         label_mask_token_id (int): token to use for masking the labels. Defaults
    #             to 2, assuming that 0 and 1 are used as the lower/higher indicator labels.
    #         aa_mask_token_id (int): token to use for masking the amino acids. Defaults
    #             to 32 (ESM alphabet).
    #         aa_mask_prob (float): probability of masking an amino acid in the embeddings
    #         label_mask_prob (float): probability of masking a label in the labels
    #         mask_both_proteins (bool): if True, we also apply a random mask to the embeddings of protein 1.
    #             This will exclude the variant positions.
    #         always_mask_variants (bool): if True, we always mask the variant positions in the 2nd protein,
    #             regardless of the mask probabilities. We only do this when the label is not masked.

    #     Note:
    #         - if the variant position is masked, the label must be unmasked (feedback from label to sequence)
    #         - if the label is masked, the variant position must not be masked (feedback from sequence to label)
    #         - never mask the variant in both labels at the same time
    #     """
        
    #     super().__init__(label_mask_token_id, aa_mask_token_id)
    #     self.aa_mask_prob = aa_mask_prob
    #     self.label_mask_prob = label_mask_prob
    #     self.mask_protein_1 = mask_both_proteins
    #     self.always_mask_variants = always_mask_variants


    # def mask(self, sequence_tokens_1: torch.Tensor, sequence_tokens_2: torch.Tensor, labels: torch.Tensor, variant_positions: Iterable[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    #     sequence_tokens_1 = sequence_tokens_1.clone()
    #     sequence_tokens_2 = sequence_tokens_2.clone()

    #     # simplify - always mask sequence 2.

    #     # 1. decide if we want to mask the labels.
    #     labels_out = labels.clone()
    #     mask_labels = torch.rand(labels.shape) < self.label_mask_prob
    #     labels_out[mask_labels] = self.label_mask_token_id

    #     label_is_masked = (labels_out[labels!=-1] == self.label_mask_token_id).any()

    #     if label_is_masked:
    #         # mode: feedback from sequence to label.
    #         # don't mask the sequence.
    #         return sequence_tokens_1, sequence_tokens_2, labels_out
    #     else:
    #         # mode: feedback from label to sequence.
    #         # mask AAs in sequence 2.

    #         mask_aa = torch.rand(sequence_tokens_2.shape[0]) < self.aa_mask_prob
    #         if self.always_mask_variants:
    #             # always mask the variant positions in the 2nd protein
    #             mask_aa[variant_positions] = 1

    #         sequence_tokens_2[mask_aa] = self.aa_mask_token_id

    #         if self.mask_protein_1:
    #             # do we want to mask in sequence 1? Only mask positions that are not the variant.
    #             mask_aa_1 = torch.rand(sequence_tokens_1.shape[0]) < self.aa_mask_prob
    #             mask_aa_1[variant_positions] = 0
    #             sequence_tokens_1[mask_aa_1] = self.aa_mask_token_id


    #         return sequence_tokens_1, sequence_tokens_2, labels_out

    

class SpanMasker(BaseMasker):
    def __init__(self,
                 label_mask_token_id: int = -100,
                 aa_mask_token_id: int = 32,
                 min_span_frac: float = 0.1,
                 max_span_frac: float = 0.15,
                 label_mask_prob: float = 0.15,
                 span_mask_prob: float = 0.15,
                 always_mask_variants: bool = False
                 ) -> None:
        """
        Mask spans of amino acids when the label is unmasked. Mask labels randomly.

        Args:
            label_mask_token_id (int): token to use for masking the labels. Defaults
                to -100
            aa_mask_token_id (int): token to use for masking the amino acids. Defaults
                to 32 (ESM alphabet).
            min_span_frac (float): minimum span length as a fraction of the sequence length
            max_span_frac (float): maximum span length as a fraction of the sequence length
            label_mask_prob (float): probability of masking a label in the labels
            span_mask_prob (float): probability of masking a span in the sequence
            always_mask_variants (bool): if True, we put a span around each variant position in the 2nd protein.
                If there is more than 1 variant, we split the span length up between the variants and mask a 
                shorter span around each variant. This is only done when the label is not masked.



        """
        super().__init__(label_mask_token_id, aa_mask_token_id)
        self.max_span_frac = max_span_frac
        self.min_span_frac = min_span_frac
        self.label_mask_prob = label_mask_prob
        self.span_mask_prob = span_mask_prob
        self.always_mask_variants = always_mask_variants

    def mask(self, sequence_tokens: torch.Tensor, label: torch.Tensor, variant_positions: Iterable[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        sequence_tokens = sequence_tokens.clone()

        # 1. decide if we want to mask the labels.
        labels_out = label.clone()
        mask_labels = torch.rand(label.shape) < self.label_mask_prob
        labels_out[mask_labels] = self.label_mask_token_id

        label_is_masked = labels_out.any()

        if label_is_masked:
            # mode: feedback from sequence to label.
            # don't mask the sequence.
            return sequence_tokens, labels_out
        else:
            # mode: feedback from label to sequence.
            # mask AAs in sequence 2.

            # sample a span length
            if self.min_span_frac == self.max_span_frac:
                span_length = int(sequence_tokens.shape[0]*self.min_span_frac)
            else:
                span_length = torch.randint(int(sequence_tokens.shape[0]*self.min_span_frac), int(sequence_tokens.shape[0]*self.max_span_frac), (1,)).item()

            # draw whether we span mask or not.
            mask_span = torch.rand(1) < self.span_mask_prob
            if not mask_span:
                return sequence_tokens, labels_out

            if self.always_mask_variants:
                # put a span around the variant positions
                span_length = max(1, span_length // len(variant_positions)) # lower bound this - can become 0 otherwise.
                for variant in variant_positions:
                    # sample a span start so that the variant is contained at a random position
                    span_start = torch.randint(variant-span_length, variant, (1,)).item()

                    span_end = span_start + span_length
                    sequence_tokens[span_start:span_end] = self.aa_mask_token_id

            else:
                # sample a span start
                span_start = torch.randint(0, sequence_tokens.shape[0]-span_length, (1,)).item()
                span_end = span_start + span_length
                sequence_tokens[span_start:span_end] = self.aa_mask_token_id

                
            return sequence_tokens, labels_out

