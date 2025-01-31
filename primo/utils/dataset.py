import torch
from torch.utils.data import IterableDataset, Dataset
from hashlib import md5
import os
import pandas as pd
import h5py
from typing import List, Dict
import numpy as np

from primo.modules.esm.data import Alphabet
from .maskers import BaseMasker
from .set_samplers import BaseSetSampler

def pad_to_min_length(seq, min_length, padding_value=1):
    """
    Pad a sequence to a minimum length.

    Args:
        seq (torch.Tensor): The sequence to pad.
        min_length (int): The minimum length.
        padding_value (int): The padding value.

    Returns:
        torch.Tensor: The padded sequence.
    """
    if seq.size(0) < min_length:
        padding_length = min_length - seq.size(0)
        return torch.nn.functional.pad(seq, (0, padding_length), value=padding_value)
    else:
        return seq


def pad_sequences(sequences, batch_first=True, padding_value=1):
    """
    Pad a list of tensors to the same length. ``utils.rnn.pad_sequence`` does not support padding our
    (set_size, L) tensors to the same length, so we need to implement our own padding function.

    Parameters:
    - sequences (list of Tensors): List of tensors to pad.
    - batch_first (bool): If True, the output tensor will have shape (batch_size, max_length, ...).
    - padding_value (float): Value to use for padding.

    Returns:
    - Tensor: Padded tensor.
    """
    # Find the maximum length
    max_length = max([seq.size(1) for seq in sequences])
    
    # Pad each sequence to the maximum length
    padded_sequences = [torch.nn.functional.pad(seq, (0, max_length - seq.size(1)), value=padding_value) for seq in sequences]
    
    # Stack the padded sequences into a single tensor
    padded_tensor = torch.stack(padded_sequences, dim=0 if batch_first else 1)
    
    return padded_tensor



class SequenceSetDataset(IterableDataset):

    def __init__(
            self,
            num_labels: int,
            num_total_sets: int,
            set_size: int,
            assay_dataframe: pd.DataFrame,
            set_sampler: BaseSetSampler,
            masker: BaseMasker,
            min_pad_length: int = None,
            is_deterministic: bool = False,
            only_mask_last: bool = False,
    ):
        """
        A dataset that samples pairs of proteins and assays and returns embeddings and labels.

        Args:
            num_labels (int): The number of labels.
            num_total_sets (int): The number of total sets to sample per epoch.
            set_size (int): The size of each set.
            assay_dataframe (pd.DataFrame): A DataFrame containing the assay data.
            pair_sampler (BasePairSampler): A pair sampler.
            masker (BaseMasker): A masker.
            is_deterministic (bool): Whether to sample pairs deterministically, repeating each epoch.
            only_mask_last (bool): Whether to only mask the last sample in each set.
        """
        
        self.num_labels = num_labels
        self.num_total_sets = num_total_sets
        self.set_size = set_size
        self.assay_dataframe = assay_dataframe

        self.aa_alphabet = Alphabet.from_architecture('ESM-1b') # alphabet for ESM2 is the same as ESM1b

        self.masker = masker
        self.set_sampler = set_sampler

        self.min_pad_length = min_pad_length
        self.is_deterministic = is_deterministic # always sample the same pairs in the same order
        self.only_mask_last = only_mask_last

        self.auxiliary_labels_mask_id = -100

    def __len__(self):
        return self.num_total_sets
    
    def __iter__(self):

        if self.is_deterministic:
            self.set_sampler.set_seed(self.set_sampler.seed)

        for i in range(self.num_total_sets):

            # sample a set
            samples = self.set_sampler.sample(self.set_size)
            # each sample is a tuple of (assay, seq, label, label_id, aux_labels, variant_positions)

            samples_torched = []
            for idx, (assay, seq, label, label_id, aux_labels, variant_positions) in enumerate(samples):
                
                # make label tensor
                label = torch.tensor(label, dtype=torch.float32)
                label_id = torch.tensor(label_id, dtype=int)
                label_target = label.clone() # this is the target for the model

                # tokenize sequences 
                # encode() does not handle bos and eos.
                if seq[0] == '.': # remove the first dot
                    seq = seq[1:]
                    bos = [self.aa_alphabet.cls_idx]
                else:
                    bos = []
                if seq[-1] == '?':
                    seq = seq[:-1]
                    eos = [self.aa_alphabet.eos_idx]
                else:
                    eos = []

                seq_tokens = self.aa_alphabet.encode(seq) # List[int]
                seq_tokens = bos + seq_tokens + eos
                seq_tokens = torch.tensor(seq_tokens, dtype=int)

                # esm: padding elements are indicated by 1s.
                pad_mask = torch.zeros(seq_tokens.shape[0], dtype=bool)

                # mask
                # last-only mode: only mask the last sample in the set.
                if self.only_mask_last and idx < self.set_size - 1:
                    masked_tokens = seq_tokens
                    labels = label.clone()
                elif self.only_mask_last and idx == self.set_size - 1:
                    masked_tokens, labels = self.masker(seq_tokens, label, variant_positions)
                # normal mode: randomly mask among all samples.
                else:
                    variant_positions = set(variant_positions)
                    masked_tokens, labels = self.masker(seq_tokens, label, variant_positions)

                # NOTE our set samplers ensure that at least the last label is different, should the preceding batch elements all have the same label.
                # in this case, for minmax batch label norm to work, ensure that we do not mask the last label.
                if  idx == self.set_size - 1 and labels == self.masker.label_mask_token_id:
                    if len(set([x[2] for x in samples[:-1]])) == 1:
                        labels = label
                    elif all([x[3] == self.masker.label_mask_token_id for x in samples_torched]):
                        # at least 1 label should be unmasked for minmax scaling to work.
                        labels = label

                    

                if self.min_pad_length is not None:
                    masked_tokens = pad_to_min_length(masked_tokens, self.min_pad_length)
                    pad_mask = pad_to_min_length(pad_mask, self.min_pad_length, padding_value=True)
                    seq_tokens = pad_to_min_length(seq_tokens, self.min_pad_length, padding_value=-1)


                # replace auxiliary label nans with mask token
                aux_labels = [self.auxiliary_labels_mask_id if np.isnan(l) else l for l in aux_labels]
                auxiliary_labels = torch.tensor(aux_labels, dtype=torch.float32)

                if torch.isnan(label).any():
                    print("nan label", label)
                    import ipdb; ipdb.set_trace()

                samples_torched.append((masked_tokens, pad_mask, label_id, labels, auxiliary_labels, seq_tokens, label_target))


            # note we need to apply collations already here, as we return a set of samples.
            masked_tokens = torch.nn.utils.rnn.pad_sequence([s[0] for s in samples_torched], batch_first=True, padding_value=1) # 1 is the padding token in the ESM alphabet.
            pad_mask = torch.nn.utils.rnn.pad_sequence([s[1] for s in samples_torched], batch_first=True, padding_value=1) # True where padded.
            labels = torch.stack([s[3] for s in samples_torched])
            label_ids = torch.stack([s[2] for s in samples_torched])
            auxiliary_labels = torch.stack([s[4] for s in samples_torched]) if samples_torched[0][4] is not None else None
            seq_tokens = torch.nn.utils.rnn.pad_sequence([s[5] for s in samples_torched], batch_first=True, padding_value=-1)
            label_target = torch.stack([s[6] for s in samples_torched])

            yield masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target


    @staticmethod
    def collate_fn(batch):

        masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target = zip(*batch)

        masked_tokens = pad_sequences(masked_tokens, batch_first=True, padding_value=1) # 1 is the padding token in the ESM alphabet.
        pad_mask = pad_sequences(pad_mask, batch_first=True, padding_value=1) # True where padded.
        labels = torch.stack(labels)
        label_ids = torch.stack(label_ids)
        auxiliary_labels = torch.stack(auxiliary_labels) if auxiliary_labels[0] is not None else None
        seq_tokens = pad_sequences(seq_tokens, batch_first=True, padding_value=-1)
        label_target = torch.stack(label_target)

        return masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target
    
    def minmax_scale_collate_fn(self, batch):
        masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target = zip(*batch)

        labels_scaled = []
        label_target_scaled = []
        # do minmax scaling of the labels for each batch element
        for i in range(len(labels)):
            # get min and max from the unmasked labels only
            min_val = labels[i][labels[i] != self.masker.label_mask_token_id].min()
            max_val = labels[i][labels[i] != self.masker.label_mask_token_id].max()

            labels_scaled.append((labels[i] - min_val) / (max_val - min_val + 1e-6) - 0.5) # center the labels
            labels_scaled[-1][labels[i] == self.masker.label_mask_token_id] = self.masker.label_mask_token_id
            label_target_scaled.append((label_target[i] - min_val) / (max_val - min_val +1e-6) - 0.5)

        masked_tokens = pad_sequences(masked_tokens, batch_first=True, padding_value=1)
        pad_mask = pad_sequences(pad_mask, batch_first=True, padding_value=1)
        # labels = torch.stack(labels)
        label_ids = torch.stack(label_ids)
        auxiliary_labels = torch.stack(auxiliary_labels) if auxiliary_labels[0] is not None else None
        seq_tokens = pad_sequences(seq_tokens, batch_first=True, padding_value=-1)
        # label_target = torch.stack(label_target)

        labels_scaled = torch.stack(labels_scaled)
        label_target_scaled = torch.stack(label_target_scaled)
        
        return masked_tokens, pad_mask, label_ids, labels_scaled, auxiliary_labels, seq_tokens, label_target_scaled


            

class TestDataset(Dataset):
    def __init__(
            self, 
            num_labels: int, 
            assay_dataframe_test: pd.DataFrame,
            masker: BaseMasker,
            conditioning_sample_inds: List[int],
            assay_dataframe_support: pd.DataFrame = None,
            min_pad_length: int = None,
            ):
        """
        A dataset for testing.

        Use this with a single-assay dataframe.

        Args:
            num_labels (int): The number of labels.
            assay_dataframe (pd.DataFrame): A DataFrame containing the assay data.
            masker (BaseMasker): A masker. We do not mask the sequences, but we need access to the label mask token.

        """

        self.num_labels = num_labels
        self.assay_dataframe_test = assay_dataframe_test
        self.assay_dataframe_support = assay_dataframe_support
        self.masker = masker
        self.min_pad_length = min_pad_length

        self.auxiliary_labels_mask_id = -100
        self.auxiliary_labels_mode = 'offset_and_difference'

        self.aa_alphabet = Alphabet.from_architecture('ESM-1b')

        self.conditioning_sample_inds = conditioning_sample_inds
        self.conditioning_samples = self._prepare_conditioning_set(conditioning_sample_inds)


    def __len__(self):
        return len(self.assay_dataframe_test)
    
    @staticmethod
    def _get_aux_labels(row: pd.Series) -> List[float]:
        """Helper function to get a list of auxiliary labels for a row."""
        aux_labels = []
        for c in row.index:
            if 'auxiliary_label' in c:
                aux_labels.append(row[c])

        return aux_labels
    

    def _torchify_one_sample(self, seq, label, label_id, aux_labels):

        # make label tensor
        label = torch.tensor(label, dtype=torch.float32)
        label_id = torch.tensor(label_id, dtype=int)

        # tokenize sequences
        if seq[0] == '.':
            seq = seq[1:]
            bos = [self.aa_alphabet.cls_idx]
        else:
            bos = []

        if seq[-1] == '?':
            seq = seq[:-1]
            eos = [self.aa_alphabet.eos_idx]
        else:
            eos = []

        seq_tokens = self.aa_alphabet.encode(seq)
        seq_tokens = bos + seq_tokens + eos
        seq_tokens = torch.tensor(seq_tokens, dtype=int)

        # esm: padding elements are indicated by 1s.
        pad_mask = torch.zeros(seq_tokens.shape[0], dtype=bool)

        # replace auxiliary label nans with mask token
        aux_labels = [self.auxiliary_labels_mask_id if np.isnan(l) else l for l in aux_labels]
        auxiliary_labels = torch.tensor(aux_labels, dtype=torch.float32)

        return seq_tokens, pad_mask, label_id, label, auxiliary_labels


    def _prepare_conditioning_set(self, conditioning_inds: List[int]):

        conditioning_df = self.assay_dataframe_support if self.assay_dataframe_support is not None else self.assay_dataframe_test

        samples_torched = []

        for idx in conditioning_inds:
            row = conditioning_df.iloc[idx]
            seq = row['mutated_sequence']
            label = row['DMS_score_modeling']
            label_id = row['property_int']
            aux_labels = self._get_aux_labels(row)

            seq_tokens, pad_mask, label_id, label, auxiliary_labels = self._torchify_one_sample(seq, label, label_id, aux_labels)
            masked_tokens = seq_tokens
            label_target = label.clone()
            labels = label.clone() # conditioning set does not have masked labels.


            samples_torched.append((masked_tokens, pad_mask, label_id, labels, auxiliary_labels, seq_tokens, label_target))


        return samples_torched




    def __getitem__(self, idx):
        
        row = self.assay_dataframe_test.iloc[idx]
        seq = row['mutated_sequence']
        label = row['DMS_score_modeling']
        label_id = row['property_int']
        aux_labels = self._get_aux_labels(row)

        seq_tokens, pad_mask, label_id, label, auxiliary_labels = self._torchify_one_sample(seq, label, label_id, aux_labels)
        masked_tokens = seq_tokens
        label_target = label.clone()
        labels = labels = torch.ones_like(label) * self.masker.label_mask_token_id

        conditioning_samples = [x for x in self.conditioning_samples] # copy

        # NOTE this is to prevent the model from cheating (same sample in support set and query set)
        # This applies only in the pure in-context setting where the support set is drawn from the same dataset (basically leave-one-out).
        # In the approach with a separate support set, we do not need this.
        if self.assay_dataframe_support is None and idx in self.conditioning_sample_inds:
            # find the dupe conditioning sample and replace it.
            bad_conditioning_sample_idx = self.conditioning_sample_inds.index(idx)
            try:
                if 'quantile' in self.assay_dataframe_test.columns:
                    quantile_df = self.assay_dataframe_test.loc[self.assay_dataframe_test['quantile'] == row['quantile']]
                    # redraw from same quantile
                    new_conditioning_sample_idx = np.random.choice([i for i in range(len(quantile_df)) if i not in self.conditioning_sample_inds + [idx]])

                else:
                    new_conditioning_sample_idx = np.random.choice([i for i in range(len(self.assay_dataframe_test)) if i not in self.conditioning_sample_inds + [idx]])

                row = self.assay_dataframe_test.iloc[new_conditioning_sample_idx]
                seq = row['mutated_sequence']
                label = row['DMS_score_modeling']
                label_id = row['property_int']
                aux_labels = self._get_aux_labels(row)
                seq_tokens, pad_mask, label_id, label, auxiliary_labels = self._torchify_one_sample(seq, label, label_id, aux_labels)
                masked_tokens = seq_tokens
                label_target = label.clone()
                labels = label.clone() # conditioning set does not have masked labels.

                conditioning_samples[bad_conditioning_sample_idx] = (masked_tokens, pad_mask, label_id, labels, auxiliary_labels, seq_tokens, label_target)
            except ValueError as e:
                print(e)
                print('Cannot resample - conditioning set is too large for this dataset.')


        samples_torched = conditioning_samples + [(masked_tokens, pad_mask, label_id, labels, auxiliary_labels, seq_tokens, label_target)]

        # note we need to apply collations already here, as we return a set of samples.
        masked_tokens = torch.nn.utils.rnn.pad_sequence([s[0] for s in samples_torched], batch_first=True, padding_value=1) # 1 is the padding token in the ESM alphabet.
        pad_mask = torch.nn.utils.rnn.pad_sequence([s[1] for s in samples_torched], batch_first=True, padding_value=1) # True where padded.
        labels = torch.stack([s[3] for s in samples_torched])
        label_ids = torch.stack([s[2] for s in samples_torched])
        auxiliary_labels = torch.stack([s[4] for s in samples_torched]) if samples_torched[0][4] is not None else None
        seq_tokens = torch.nn.utils.rnn.pad_sequence([s[5] for s in samples_torched], batch_first=True, padding_value=-1)
        label_target = torch.stack([s[6] for s in samples_torched])

        


        return masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target



    @staticmethod
    def collate_fn(batch):

        masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target = zip(*batch)

        masked_tokens = pad_sequences(masked_tokens, batch_first=True, padding_value=1) # 1 is the padding token in the ESM alphabet.
        pad_mask = pad_sequences(pad_mask, batch_first=True, padding_value=1) # True where padded.
        labels = torch.stack(labels)
        label_ids = torch.stack(label_ids)
        auxiliary_labels = torch.stack(auxiliary_labels) if auxiliary_labels[0] is not None else None
        seq_tokens = pad_sequences(seq_tokens, batch_first=True, padding_value=-1)
        label_target = torch.stack(label_target)

        return masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target


    def minmax_scale_collate_fn(self, batch):
        masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target = zip(*batch)

        labels_scaled = []
        label_target_scaled = []
        # do minmax scaling of the labels for each batch element
        for i in range(len(labels)):
            # get min and max from the unmasked labels only
            min_val = labels[i][labels[i] != self.masker.label_mask_token_id].min()
            max_val = labels[i][labels[i] != self.masker.label_mask_token_id].max()

            labels_scaled.append((labels[i] - min_val) / (max_val - min_val + 1e-6) - 0.5) # center the labels  
            labels_scaled[-1][labels[i] == self.masker.label_mask_token_id] = self.masker.label_mask_token_id
            label_target_scaled.append((label_target[i] - min_val) / (max_val - min_val +1e-6) - 0.5)

        masked_tokens = pad_sequences(masked_tokens, batch_first=True, padding_value=1)
        pad_mask = pad_sequences(pad_mask, batch_first=True, padding_value=1)
        # labels = torch.stack(labels)
        label_ids = torch.stack(label_ids)
        auxiliary_labels = torch.stack(auxiliary_labels) if auxiliary_labels[0] is not None else None
        seq_tokens = pad_sequences(seq_tokens, batch_first=True, padding_value=-1)
        # label_target = torch.stack(label_target)

        labels_scaled = torch.stack(labels_scaled)
        label_target_scaled = torch.stack(label_target_scaled) 

        return masked_tokens, pad_mask, label_ids, labels_scaled, auxiliary_labels, seq_tokens, label_target_scaled



class SupportDataframeTestDataset(Dataset):
    def __init__(
            self, 
            num_labels: int, 
            assay_dataframe: pd.DataFrame,
            masker: BaseMasker,
            conditioning_dataframe: pd.DataFrame,
            min_pad_length: int = None,
            ):
        """
        A dataset for testing.
        Use this with a single-assay dataframe. Instead of drawing conditioning samples from the same dataset, we condition
        on a separate dataframe.



        Args:
            num_labels (int): The number of labels.
            assay_dataframe (pd.DataFrame): A DataFrame containing the assay data.
            masker (BaseMasker): A masker. We do not mask the sequences, but we need access to the label mask token.

        """

        self.num_labels = num_labels
        self.assay_dataframe = assay_dataframe
        self.masker = masker
        self.min_pad_length = min_pad_length

        self.auxiliary_labels_mask_id = -100
        self.auxiliary_labels_mode = 'offset_and_difference'

        self.aa_alphabet = Alphabet.from_architecture('ESM-1b')

        self.conditioning_dataframe = conditioning_dataframe
        self.conditioning_samples = self._prepare_conditioning_set(conditioning_dataframe)


    def __len__(self):
        return len(self.assay_dataframe)
    
    @staticmethod
    def _get_aux_labels(row: pd.Series) -> List[float]:
        """Helper function to get a list of auxiliary labels for a row."""
        aux_labels = []
        for c in row.index:
            if 'auxiliary_label' in c:
                aux_labels.append(row[c])

        return aux_labels
    

    def _torchify_one_sample(self, seq, label, label_id, aux_labels):

        # make label tensor
        label = torch.tensor(label, dtype=torch.float32)
        label_id = torch.tensor(label_id, dtype=int)

        # tokenize sequences
        if seq[0] == '.':
            seq = seq[1:]
            bos = [self.aa_alphabet.cls_idx]
        else:
            bos = []

        if seq[-1] == '?':
            seq = seq[:-1]
            eos = [self.aa_alphabet.eos_idx]
        else:
            eos = []

        seq_tokens = self.aa_alphabet.encode(seq)
        seq_tokens = bos + seq_tokens + eos
        seq_tokens = torch.tensor(seq_tokens, dtype=int)

        # esm: padding elements are indicated by 1s.
        pad_mask = torch.zeros(seq_tokens.shape[0], dtype=bool)

        # replace auxiliary label nans with mask token
        aux_labels = [self.auxiliary_labels_mask_id if np.isnan(l) else l for l in aux_labels]
        auxiliary_labels = torch.tensor(aux_labels, dtype=torch.float32)

        return seq_tokens, pad_mask, label_id, label, auxiliary_labels


    def _prepare_conditioning_set(self, conditioning_dataframe: pd.DataFrame):

        samples_torched = []

        for idx, row in conditioning_dataframe.iterrows():
            seq = row['mutated_sequence']
            label = row['DMS_score_modeling']
            label_id = row['property_int']
            aux_labels = self._get_aux_labels(row)

            seq_tokens, pad_mask, label_id, label, auxiliary_labels = self._torchify_one_sample(seq, label, label_id, aux_labels)
            masked_tokens = seq_tokens
            label_target = label.clone()
            labels = label.clone() # conditioning set does not have masked labels.


            samples_torched.append((masked_tokens, pad_mask, label_id, labels, auxiliary_labels, seq_tokens, label_target))


        return samples_torched




    def __getitem__(self, idx):
        
        row = self.assay_dataframe.iloc[idx]
        seq = row['mutated_sequence']
        label = row['DMS_score_modeling']
        label_id = row['property_int']
        aux_labels = self._get_aux_labels(row)

        seq_tokens, pad_mask, label_id, label, auxiliary_labels = self._torchify_one_sample(seq, label, label_id, aux_labels)
        masked_tokens = seq_tokens
        label_target = label.clone()
        labels = labels = torch.ones_like(label) * self.masker.label_mask_token_id

        conditioning_samples = [x for x in self.conditioning_samples] # copy
        samples_torched = conditioning_samples + [(masked_tokens, pad_mask, label_id, labels, auxiliary_labels, seq_tokens, label_target)]

        # note we need to apply collations already here, as we return a set of samples.
        masked_tokens = torch.nn.utils.rnn.pad_sequence([s[0] for s in samples_torched], batch_first=True, padding_value=1) # 1 is the padding token in the ESM alphabet.
        pad_mask = torch.nn.utils.rnn.pad_sequence([s[1] for s in samples_torched], batch_first=True, padding_value=1) # True where padded.
        labels = torch.stack([s[3] for s in samples_torched])
        label_ids = torch.stack([s[2] for s in samples_torched])
        auxiliary_labels = torch.stack([s[4] for s in samples_torched]) if samples_torched[0][4] is not None else None
        seq_tokens = torch.nn.utils.rnn.pad_sequence([s[5] for s in samples_torched], batch_first=True, padding_value=-1)
        label_target = torch.stack([s[6] for s in samples_torched])

        
        return masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target



    @staticmethod
    def collate_fn(batch):

        masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target = zip(*batch)

        masked_tokens = pad_sequences(masked_tokens, batch_first=True, padding_value=1) # 1 is the padding token in the ESM alphabet.
        pad_mask = pad_sequences(pad_mask, batch_first=True, padding_value=1) # True where padded.
        labels = torch.stack(labels)
        label_ids = torch.stack(label_ids)
        auxiliary_labels = torch.stack(auxiliary_labels) if auxiliary_labels[0] is not None else None
        seq_tokens = pad_sequences(seq_tokens, batch_first=True, padding_value=-1)
        label_target = torch.stack(label_target)

        return masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target


    def minmax_scale_collate_fn(self, batch):
        masked_tokens, pad_mask, label_ids, labels, auxiliary_labels, seq_tokens, label_target = zip(*batch)

        labels_scaled = []
        label_target_scaled = []
        # do minmax scaling of the labels for each batch element
        for i in range(len(labels)):
            # get min and max from the unmasked labels only
            min_val = labels[i][labels[i] != self.masker.label_mask_token_id].min()
            max_val = labels[i][labels[i] != self.masker.label_mask_token_id].max()

            labels_scaled.append((labels[i] - min_val) / (max_val - min_val + 1e-6) - 0.5) # center the labels  
            labels_scaled[-1][labels[i] == self.masker.label_mask_token_id] = self.masker.label_mask_token_id
            label_target_scaled.append((label_target[i] - min_val) / (max_val - min_val +1e-6) - 0.5)

        masked_tokens = pad_sequences(masked_tokens, batch_first=True, padding_value=1)
        pad_mask = pad_sequences(pad_mask, batch_first=True, padding_value=1)
        # labels = torch.stack(labels)
        label_ids = torch.stack(label_ids)
        auxiliary_labels = torch.stack(auxiliary_labels) if auxiliary_labels[0] is not None else None
        seq_tokens = pad_sequences(seq_tokens, batch_first=True, padding_value=-1)
        # label_target = torch.stack(label_target)

        labels_scaled = torch.stack(labels_scaled)
        label_target_scaled = torch.stack(label_target_scaled) 

        return masked_tokens, pad_mask, label_ids, labels_scaled, auxiliary_labels, seq_tokens, label_target_scaled
    

    ## For baseline models


class BaselineSequenceDataset(Dataset):

    def __init__(
            self,
            num_labels: int,
            assay_dataframe: pd.DataFrame,
            min_pad_length: int = None,
    ):
        """
        A dataset that returns embeddings and labels.

        Args:
            num_labels (int): The number of labels.
            assay_dataframe (pd.DataFrame): A DataFrame containing the assay data.
        """
        
        self.num_labels = num_labels
        self.assay_dataframe = assay_dataframe

        self.aa_alphabet = Alphabet.from_architecture('ESM-1b') # alphabet for ESM2 is the same as ESM1b


        self.min_pad_length = min_pad_length
        self.auxiliary_labels_mask_id = 0 # TODO work out how to handle this. #-100

    def __len__(self):
        return len(self.assay_dataframe)
    
    def _get_aux_labels(self, row: pd.Series) -> List[float]:
        """Helper function to get a list of auxiliary labels for a row."""
        aux_labels = []
        for c in row.index:
            if 'auxiliary_label' in c:
                aux_labels.append(row[c])

        return aux_labels
    
    def __getitem__(self, idx):

        row = self.assay_dataframe.iloc[idx]

        # TODO check this
        seq, label = row['mutated_sequence'], row['DMS_score_modeling']
        auxiliary_labels = self._get_aux_labels(row)

         # make label tensor
        label = torch.tensor(label, dtype=torch.float32)

        bos = [self.aa_alphabet.cls_idx]
        eos = [self.aa_alphabet.eos_idx]


        seq_tokens = self.aa_alphabet.encode(seq) # List[int]
        seq_tokens = bos + seq_tokens + eos
        seq_tokens = torch.tensor(seq_tokens, dtype=int)

        # esm: padding elements are indicated by 1s.
        pad_mask = torch.zeros(seq_tokens.shape[0], dtype=bool)

            
        if self.min_pad_length is not None:
            pad_mask = pad_to_min_length(pad_mask, self.min_pad_length, padding_value=True)
            seq_tokens = pad_to_min_length(seq_tokens, self.min_pad_length, padding_value=1) # esm pad token=1


        # replace auxiliary label nans with mask token
        auxiliary_labels = [self.auxiliary_labels_mask_id if np.isnan(l) else l for l in auxiliary_labels]
        auxiliary_labels = torch.tensor(auxiliary_labels, dtype=torch.float32)

        if torch.isnan(label).any():
            print("nan label", label)
            import ipdb; ipdb.set_trace()



        return seq_tokens, pad_mask, auxiliary_labels, label

    @staticmethod
    def collate_fn(batch):

        seq_tokens, pad_mask, auxiliary_labels, label_target = zip(*batch)


        seq_tokens = torch.nn.utils.rnn.pad_sequence(seq_tokens, batch_first=True, padding_value=1) # esm pad token=1
        pad_mask = torch.nn.utils.rnn.pad_sequence(pad_mask, batch_first=True, padding_value=1)

        # seq_tokens = pad_sequences(seq_tokens, batch_first=True, padding_value=-1)
        # pad_mask = pad_sequences(pad_mask, batch_first=True, padding_value=1) # True where padded.
        auxiliary_labels = torch.stack(auxiliary_labels) if auxiliary_labels[0] is not None else None
        label_target = torch.stack(label_target)

        return seq_tokens, pad_mask, auxiliary_labels, label_target
    
            
