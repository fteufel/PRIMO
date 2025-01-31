import torch
from typing import Tuple, Any, List, Optional, Dict

from abc import ABC, ABCMeta, abstractmethod
import pandas as pd
import numpy as np


class BaseSetSampler(ABC):
        
    def __init__(
            self, 
            dataframe: pd.DataFrame, 
            seed=123, 
            add_special_tokens: bool = False, 
            dummy_bos_symbol: str = '.', 
            dummy_eos_symbol: str='?'
            ) -> None:
        """
        Base class for pair samplers.

        Args:
            dataframe (pd.DataFrame): A DataFrame containing the assay data.
                The DataFrame should have the following columns:
                - assay: The assay name.
                - mutated_sequence: The mutated sequence.
                - DMS_score: float The DMS score.
                - property_int: int The property integer.
                - DMS_score_bin: int The DMS score bin.

            seed (int): Seed for the random number generator.
            add_special_tokens (bool): Add special tokens to the sequences. This helps with cropping while maintaining EOS and BOS tokens.
            dummy_bos_symbol (str): The dummy BOS symbol.
            dummy_eos_symbol (str): The dummy EOS symbol.
        """
        dataframe = dataframe.reset_index(drop=True)
        # dataframe = dataframe.set_index('assay')
        dataframe = dataframe.set_index('assay', append=True).reorder_levels([1,0]).sort_index()

        if add_special_tokens:
            dataframe['mutated_sequence'] = dummy_bos_symbol + dataframe['mutated_sequence'] + dummy_eos_symbol
            dataframe['mutant_range_start'] = dataframe['mutant_range_start'] + 1
            dataframe['mutant_range_end'] = dataframe['mutant_range_end'] + 1

        # make the dataframe easier to work with. expand lists to columns.
        # find all properties that exist in the dataset
        all_properties = set()
        for idx, row in dataframe.iterrows():
            all_properties.update([row['property_int']])
        # make colums for each property.
        self.all_properties = sorted(list(all_properties))

        self.dataframe = dataframe
        self.seed = seed
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.seed = seed + worker_info.id
        self.rng = np.random.RandomState(self.seed)

    @abstractmethod
    def sample(self, set_size: int, ignore_inds: List[int] = None, assay: str = None) -> List[Tuple]:
        pass

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, *args: Any, **kwargs: Any):
            return self.sample(*args, **kwargs)


    def _get_aux_labels(self, row: pd.Series) -> List[float]:
        """Helper function to get a list of auxiliary labels for a row."""
        aux_labels = []
        for c in row.index:
            if 'auxiliary_label' in c:
                aux_labels.append(row[c])

        return aux_labels



class BaseSubsetter(ABC):
    """
    Subsetters implement an API to subset a DataFrame based on some criteria.
    
    We could achieve the same with a function, but this allows us to also have
    static arguments that are set at initialization, and to have a consistent API.

    To implement a sampling approach, we chain subsetters and random sampling steps.
    """

    @abstractmethod
    def subset(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """Subset a DataFrame based on some criteria."""
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.subset(*args, **kwds)
    

class AssaySubsetter(BaseSubsetter):
    def __init__(self) -> None:
        """A subsetter that subsets a DataFrame based on the assay name"""

    def subset(self, df: pd.DataFrame, assay: str) -> pd.DataFrame:
        """Subset a DataFrame based on the assay name."""
        return df.loc[assay]

class StandardDeviationSubsetter(BaseSubsetter):
    def __init__(self, stdev_mult: float, stdevs: Dict[str, List[float]]) -> None:
        """A subsetter that subsets a DataFrame based on the standard deviation of the DMS scores."""
        self.stdev_mult: float = stdev_mult
        self.stdevs: Dict[str, List[float]] = stdevs

    def subset(self, df: pd.DataFrame, reference_score: float, assay: str) -> pd.DataFrame:
        """Subset a DataFrame based on the standard deviation of the DMS scores."""
        stdev = self.stdevs[assay]
        return df[(df[f'DMS_score'] - reference_score).abs() >= self.stdev_mult * stdev]

class BimodalSubsetter(BaseSubsetter):
    def __init__(self, ) -> None:
        """A subsetter that subsets a DataFrame based on a precomputed bisection of the DMS scores."""

    @staticmethod
    def subset(df: pd.DataFrame, mode: int) -> pd.DataFrame:
        """Subset a DataFrame based on the mode of the DMS scores."""
        return df[df[f'DMS_score_bin'] == mode]

class NotSameValueSubsetter(BaseSubsetter):
    def __init__(self, ) -> None:
        """A subsetter that subsets a DataFrame based on excluding a decile of the DMS scores."""

    @staticmethod
    def subset(df: pd.DataFrame, value: int) -> pd.DataFrame:
        """Subset a DataFrame based on the value of the DMS scores."""
        return df[df[f'DMS_score_modeling'] != value]


    
class MutantWindowSubsetter(BaseSubsetter):
    def __init__(self, window_size: int) -> None:
        """A subsetter that subsets a DataFrame based on the mutant range of the sequences."""
        self.window_size = window_size

    def subset(self, df: pd.DataFrame, start_position: int, end_position: int) -> pd.DataFrame:
        """Subset a DataFrame based on the mutant range of the sequences."""

        return df[(df['mutant_range_start'] >= start_position) & (df['mutant_range_end'] <= end_position)]
    

class RestartWhileLoop(Exception):
    pass
    
class SimpleSetSampler(BaseSetSampler):
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        seed=123, 
        window_size: int = None,
        add_special_tokens: bool = False,
        return_variant_positions: bool = False,
        return_aux_labels: bool = False,
        property_sample_mode = 'assay_random',
        ) -> None:
        """
        Samples sets that
        - are from the same assay
        - have their mutant range within the window size
        - contain at least 2 different DMS values
        """

        super().__init__(dataframe, seed, add_special_tokens)

        # prefilter the dataframe so that only mutants that fit the window size are included
        if window_size is not None:
            self.dataframe = self.dataframe[self.dataframe['mutant_range_end'] - self.dataframe['mutant_range_start'] <= window_size]

        self.assays = self.dataframe.index.get_level_values(0).unique()
        self.assay_properties = self.dataframe.reset_index().groupby('property_int')['assay'].unique().to_dict()

        self.window_size = window_size
        self.return_variant_positions = return_variant_positions
        self.return_aux_labels = return_aux_labels

        self.properties = self.dataframe['property_int'].unique()
        self.property_sample_mode = property_sample_mode

        if self.property_sample_mode == 'assay_random':
            # sample each property proportional to its number of unique assays
            assay_counts = self.dataframe.groupby('property_int').apply(lambda df: df.index.get_level_values(0).nunique())
            self.property_p = assay_counts / assay_counts.sum()
        elif self.property_sample_mode == 'assay_sqrt':
            # sample each property proportional to the square root of its number of unique assays
            assay_counts = self.dataframe.groupby('property_int').apply(lambda df: df.index.get_level_values(0).nunique())
            self.property_p = assay_counts.apply(np.sqrt)
            self.property_p /= self.property_p.sum()
        elif self.property_sample_mode == 'random':
            self.property_p = None


        self.assay_subsetter = AssaySubsetter()
        self.value_subsetter = NotSameValueSubsetter()
        self.mutant_window_subsetter = MutantWindowSubsetter(window_size)




    def _draw_valid_window(self, seq1_mut_start, seq1_mut_end, seq1_len):

        allowed_min_start_position = max(0, seq1_mut_end - self.window_size) # so that if we start at this position, we will include the mutant range
        allowed_max_start_position = min(seq1_mut_start, max(0,seq1_len - self.window_size)) # so that if we start at this position, we will include the full protein if it is shorter than window_size

        if allowed_max_start_position == 0:
            start_position = 0
        else:
            start_position = self.rng.randint(allowed_min_start_position, allowed_max_start_position+1) # NOTE high is exclusive

        end_position = start_position + self.window_size

        return start_position, end_position

    def sample(self, set_size: int = 5, ignore_inds: List[int] = None, assay: str = None, return_orig_inds: bool=False) -> Tuple[str, str, str, str, int, int]:

        if ignore_inds is None:
            ignore_inds = []
            
        assay_in = assay
        assay = None
        
        tries = 0
        found = False
        # while tries < 100 and found == False:
        df = self.dataframe.copy()

        # drop ignore indices - these are .iloc indices
        df = df.iloc[[x for x in range(len(df)) if x not in ignore_inds]]

        # 1. draw an assay
        if assay_in is None:
            # draw a property
            property_int = self.rng.choice(self.properties, p=self.property_p)
            # draw an assay with that property
            assays_with_property = self.assay_properties[property_int]
            assay = self.rng.choice(assays_with_property)
        else:
            assay = assay_in
        df_assay = self.assay_subsetter.subset(df, assay)

        # NOTE moved this here to trigger failure when we cannot sample from an assay - all assays need to be sampleable.
        while tries < 2000 and found == False:

            if len(df_assay) < 1:
                print(assay, tries)
                import ipdb; ipdb.set_trace()
                print('Error: Could not sample from assay. Exiting.')
                exit()

            # 2. draw sequence
            seq1_idx_int = self.rng.randint(0, len(df_assay))
            seq1_idx = df_assay.iloc[seq1_idx_int].name
        
            # 3. draw the window containing sequence_1 mutant range
            seq1_mut_start = self.dataframe.loc[assay, seq1_idx]['mutant_range_start']
            seq1_mut_end = self.dataframe.loc[assay, seq1_idx]['mutant_range_end']
            seq1_len = len(self.dataframe.loc[assay, seq1_idx]['mutated_sequence'])
            if self.window_size is not None:
                start_position, end_position = self._draw_valid_window(seq1_mut_start, seq1_mut_end, seq1_len)
                # 4. subset the data to only include sequences that have their mutant range within the window
                df = self.mutant_window_subsetter.subset(df_assay, start_position, end_position)
            else:
                df = df_assay
            if len(df) == 0:
                # if we have 1 sample selected and the df is empty already,
                # we need to retry, and pick something else to start with
                # to get a valid window.
                tries += 1
                continue


            samples = [seq1_idx]
            # find the rest of the samples in the window
            # wrapped in try-except to restart the while loop when a for loop iteration fails
            try:
                for i in range(set_size-1):

                    if i == set_size-2 and len(set([self.dataframe.loc[assay, x]['DMS_score_modeling'] for x in samples])) == 1:
                    # ensure that there are at least 2 different values in the set
                        df = self.value_subsetter.subset(df, self.dataframe.loc[assay, seq1_idx]['DMS_score_modeling'])
                        if len(df) == 0:
                            tries += 1
                            raise RestartWhileLoop


                    seq2_idx_int = self.rng.randint(0, len(df))
                    seq2_idx = df.iloc[seq2_idx_int].name
                    samples.append(seq2_idx)
            except RestartWhileLoop:
                continue

            found = True

        if found == False:
            raise ValueError(f'Could not find a set with window_size {self.window_size} in asay {assay} in 2000 attempts.')

        # prepare outputs
        outs = []
        for seq_idx in samples:
            seq = self.dataframe.loc[assay, seq_idx]['mutated_sequence'] if self.window_size is None else self.dataframe.loc[assay, seq_idx]['mutated_sequence'][start_position:end_position]
            label = self.dataframe.loc[assay, seq_idx]['DMS_score_modeling']
            label_id = self.dataframe.loc[assay, seq_idx]['property_int']
            aux_labels = self._get_aux_labels(self.dataframe.loc[assay, seq_idx])

            out = (assay, seq, label, label_id, aux_labels)
            if self.return_variant_positions:
                variant_positions = self.dataframe.loc[assay, seq_idx]['mutant_positions']
                out += (variant_positions,)
            outs.append(out)

        if return_orig_inds:
            # find the index in self.dataframe for each sample
            orig_indices = [self.dataframe.index.get_loc((assay, x)) for x in samples]
            return outs, orig_indices

        return outs


class QuantileStratifiedSetSampler(BaseSetSampler):
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        seed=123, 
        window_size: int = None,
        add_special_tokens: bool = False,
        return_variant_positions: bool = False,
        return_aux_labels: bool = False,
        ) -> None:
        """
        Samples sets that
        - are from the same assay
        - have 1 sample from each quantile of the DMS scores (set size = number of quantiles)
        """

        super().__init__(dataframe, seed, add_special_tokens)

        # prefilter the dataframe so that only mutants that fit the window size are included
        self.dataframe = self.dataframe[self.dataframe['mutant_range_end'] - self.dataframe['mutant_range_start'] <= window_size]

        self.assays = self.dataframe.index.get_level_values(0).unique()

        self.window_size = window_size
        self.return_variant_positions = return_variant_positions
        self.return_aux_labels = return_aux_labels

        self.assay_subsetter = AssaySubsetter()
        self.value_subsetter = NotSameValueSubsetter()
        self.mutant_window_subsetter = MutantWindowSubsetter(window_size)




    def _draw_valid_window(self, seq1_mut_start, seq1_mut_end, seq1_len):

        allowed_min_start_position = max(0, seq1_mut_end - self.window_size) # so that if we start at this position, we will include the mutant range
        allowed_max_start_position = min(seq1_mut_start, max(0,seq1_len - self.window_size)) # so that if we start at this position, we will include the full protein if it is shorter than window_size

        if allowed_max_start_position == 0:
            start_position = 0
        else:
            start_position = self.rng.randint(allowed_min_start_position, allowed_max_start_position+1) # NOTE high is exclusive

        end_position = start_position + self.window_size

        return start_position, end_position

    def sample(self, set_size: int = 5, ignore_inds: List[int] = None, assay: str = None, return_orig_inds: bool = False) -> Tuple[str, str, str, str, int, int]:

        if ignore_inds is None:
            ignore_inds = []

        # hacky thing to allow for the assay to be passed in, but preserving
        # previous usage with resampling
        assay_in = assay
        assay = None

        tries = 0
        found = False
        while tries < 100 and found == False:

            df = self.dataframe.copy()
            # drop ignore indices - these are .iloc indices
            df = df.iloc[[x for x in range(len(df)) if x not in ignore_inds]]

            # 1. draw an assay
            if assay_in is None:
                assay = self.rng.choice(self.assays)
            else:
                assay = assay_in
        
            # 1. draw an assay
            df = self.assay_subsetter.subset(df, assay)

            # add quantiles to the dataframe
            df = df.copy()
            df['quantile'] = pd.qcut(df['DMS_score_modeling'], set_size, labels=False, duplicates='drop')

            # # 2. draw sequence from quantile 0
            try:
                seq_1_idx_int = self.rng.choice(df[df['quantile'] == 0].index)
            except ValueError: # this can fail if we have a single-value assay
                tries += 1
                continue

            seq1_idx = df.loc[seq_1_idx_int].name
        
            # # 3. draw the window containing sequence_1 mutant range
            seq1_mut_start = self.dataframe.loc[assay, seq1_idx]['mutant_range_start']
            seq1_mut_end = self.dataframe.loc[assay, seq1_idx]['mutant_range_end']
            seq1_len = len(self.dataframe.loc[assay, seq1_idx]['mutated_sequence'])
            start_position, end_position = self._draw_valid_window(seq1_mut_start, seq1_mut_end, seq1_len)

            # # 4. subset the data to only include sequences that have their mutant range within the window
            df = self.mutant_window_subsetter.subset(df, start_position, end_position)
            if len(df) == 0:
                tries += 1
                continue


            samples = [seq1_idx]
            # find the rest of the samples in the window
            # wrapped in try-except to restart the while loop when a for loop iteration fails
            try:
                for i in range(1, set_size):

                    if i == set_size-2 and len(set([self.dataframe.loc[assay, x]['DMS_score_modeling'] for x in samples])) == 1:
                    # ensure that there are at least 2 different values in the set
                        df = self.value_subsetter.subset(df, self.dataframe.loc[assay, seq1_idx]['DMS_score_modeling'])
                        if len(df) == 0:
                            tries += 1
                            raise RestartWhileLoop

                    # draw from the quantile i
                    quantile_df = df[df['quantile'] == i]
                    if len(quantile_df) == 0:
                        tries += 1
                        raise RestartWhileLoop
                    seq2_idx_int = self.rng.choice(quantile_df.index)
                    seq2_idx = df.loc[seq2_idx_int].name
                    samples.append(seq2_idx)
            except RestartWhileLoop:
                continue

            found = True

        if found == False:
            import ipdb; ipdb.set_trace()
            raise ValueError(f'Could not find a set with window_size {self.window_size} in 100 attempts.')

        # prepare outputs
        outs = []
        for seq_idx in samples:
            seq = self.dataframe.loc[assay, seq_idx]['mutated_sequence'][start_position:end_position]
            label = self.dataframe.loc[assay, seq_idx]['DMS_score_modeling']
            label_id = self.dataframe.loc[assay, seq_idx]['property_int']
            aux_labels = self._get_aux_labels(self.dataframe.loc[assay, seq_idx])

            out = (assay, seq, label, label_id, aux_labels)
            if self.return_variant_positions:
                variant_positions = self.dataframe.loc[assay, seq_idx]['mutant_positions']
                out += (variant_positions,)
            outs.append(out)

        if return_orig_inds:
            # find the index in self.dataframe for each sample
            orig_indices = [self.dataframe.index.get_loc((assay, x)) for x in samples]
            return outs, orig_indices

        return outs

class QuantileStratifiedPlusKSetSampler(BaseSetSampler):
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        seed=123, 
        window_size: int = None,
        add_special_tokens: bool = False,
        return_variant_positions: bool = False,
        return_aux_labels: bool = False,
        k: int = 1,
        ) -> None:
        """
        Samples sets that
        - are from the same assay
        - have 1 sample from each quantile of the DMS scores (set size = number of quantiles)
        """

        super().__init__(dataframe, seed, add_special_tokens)

        # prefilter the dataframe so that only mutants that fit the window size are included
        self.dataframe = self.dataframe[self.dataframe['mutant_range_end'] - self.dataframe['mutant_range_start'] <= window_size]

        self.assays = self.dataframe.index.get_level_values(0).unique()

        self.k = k # the number of random samples to add to the set

        self.window_size = window_size
        self.return_variant_positions = return_variant_positions
        self.return_aux_labels = return_aux_labels

        self.assay_subsetter = AssaySubsetter()
        self.value_subsetter = NotSameValueSubsetter()
        self.mutant_window_subsetter = MutantWindowSubsetter(window_size)




    def _draw_valid_window(self, seq1_mut_start, seq1_mut_end, seq1_len):

        allowed_min_start_position = max(0, seq1_mut_end - self.window_size) # so that if we start at this position, we will include the mutant range
        allowed_max_start_position = min(seq1_mut_start, max(0,seq1_len - self.window_size)) # so that if we start at this position, we will include the full protein if it is shorter than window_size

        if allowed_max_start_position == 0:
            start_position = 0
        else:
            start_position = self.rng.randint(allowed_min_start_position, allowed_max_start_position+1) # NOTE high is exclusive

        end_position = start_position + self.window_size

        return start_position, end_position

    def sample(self, set_size: int = 5, ignore_inds: List[int] = None, assay: str = None, return_orig_inds: bool = False) -> Tuple[str, str, str, str, int, int]:

        set_size = set_size - self.k

        if ignore_inds is None:
            ignore_inds = []

        # hacky thing to allow for the assay to be passed in, but preserving
        # previous usage with resampling
        assay_in = assay
        assay = None

        tries = 0
        found = False
        while tries < 100 and found == False:

            df = self.dataframe.copy()
            # drop ignore indices - these are .iloc indices
            df = df.iloc[[x for x in range(len(df)) if x not in ignore_inds]]

            # 1. draw an assay
            if assay_in is None:
                assay = self.rng.choice(self.assays)
            else:
                assay = assay_in
        
            # 1. draw an assay
            df = self.assay_subsetter.subset(df, assay)

            # add quantiles to the dataframe
            df = df.copy()
            df['quantile'] = pd.qcut(df['DMS_score_modeling'], set_size, labels=False, duplicates='drop')

            # # 2. draw sequence from quantile 0
            print(df['quantile'].unique())
            try:
                seq_1_idx_int = self.rng.choice(df[df['quantile'] == 0].index)
            except ValueError: # this can fail if we have a single-value assay
                tries += 1
                continue

            seq1_idx = df.loc[seq_1_idx_int].name
        
            # # 3. draw the window containing sequence_1 mutant range
            seq1_mut_start = self.dataframe.loc[assay, seq1_idx]['mutant_range_start']
            seq1_mut_end = self.dataframe.loc[assay, seq1_idx]['mutant_range_end']
            seq1_len = len(self.dataframe.loc[assay, seq1_idx]['mutated_sequence'])
            start_position, end_position = self._draw_valid_window(seq1_mut_start, seq1_mut_end, seq1_len)

            # # 4. subset the data to only include sequences that have their mutant range within the window
            df = self.mutant_window_subsetter.subset(df, start_position, end_position)
            if len(df) == 0:
                tries += 1
                continue


            samples = [seq1_idx]
            # find the rest of the samples in the window
            # wrapped in try-except to restart the while loop when a for loop iteration fails
            try:
                for i in range(1, set_size):

                    if i == set_size-2 and len(set([self.dataframe.loc[assay, x]['DMS_score_modeling'] for x in samples])) == 1:
                    # ensure that there are at least 2 different values in the set
                        df = self.value_subsetter.subset(df, self.dataframe.loc[assay, seq1_idx]['DMS_score_modeling'])
                        if len(df) == 0:
                            tries += 1
                            raise RestartWhileLoop

                    # draw from the quantile i
                    quantile_df = df[df['quantile'] == i]
                    if len(quantile_df) == 0:
                        tries += 1
                        raise RestartWhileLoop
                    seq2_idx_int = self.rng.choice(quantile_df.index)
                    seq2_idx = df.loc[seq2_idx_int].name
                    samples.append(seq2_idx)
            except RestartWhileLoop:
                continue

            # add the k randoms
            # remove already drawn samples
            df = df.loc[~df.index.isin(samples)]
            if len(df) < self.k:
                tries += 1
                continue
            for i in range(self.k):
                seq2_idx_int = self.rng.choice(df.index)
                seq2_idx = df.loc[seq2_idx_int].name
                samples.append(seq2_idx)

            found = True

        if found == False:
            import ipdb; ipdb.set_trace()
            raise ValueError(f'Could not find a set with window_size {self.window_size} in 100 attempts.')

        # prepare outputs
        outs = []
        for seq_idx in samples:
            seq = self.dataframe.loc[assay, seq_idx]['mutated_sequence'][start_position:end_position]
            label = self.dataframe.loc[assay, seq_idx]['DMS_score_modeling']
            label_id = self.dataframe.loc[assay, seq_idx]['property_int']
            aux_labels = self._get_aux_labels(self.dataframe.loc[assay, seq_idx])

            out = (assay, seq, label, label_id, aux_labels)
            if self.return_variant_positions:
                variant_positions = self.dataframe.loc[assay, seq_idx]['mutant_positions']
                out += (variant_positions,)
            outs.append(out)

        if return_orig_inds:
            # find the index in self.dataframe for each sample
            orig_indices = [self.dataframe.index.get_loc((assay, x)) for x in samples]
            return outs, orig_indices

        return outs




class CombinedProbabilitySampler():
    def __init__(self, samplers: List[BaseSetSampler], weights: List[float], seed=123) -> None:
        """Samples sets from a list of samplers with given weights."""
        super().__init__()
        assert len(samplers) == len(weights), 'Number of weights must match number of samplers.'
        self.samplers = samplers
        self.weights = weights

        self.rng = np.random.RandomState(seed)

    def set_weights(self, weights: List[float]) -> None:
        assert len(weights) == len(self.samplers), 'Number of weights must match number of samplers.'
        self.weights = weights

    def sample(self):

        sampler = self.rng.choice(self.samplers, p=self.weights)
        return sampler.sample()
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.sample(*args, **kwds)

class CombinedSetSampler():
    def __init__(self, samplers: List[BaseSetSampler], sizes: List[int]) -> None:
        """Samples a set by concatenating sets from multiple samplers."""
        assert len(samplers) == len(sizes), 'Number of sizes must match number of samplers.'
        self.samplers = samplers
        self.sizes = sizes
        self.defined_size = sum(sizes)

    def sample(self, size: int) -> List[Tuple]:

        if size != self.defined_size:
            raise ValueError(f'Size must be equal to the combined size of the samplers: {self.defined_size}')
        
        out_set = []
        assay = None
        ignore_inds = []
        for sampler, size in zip(self.samplers, self.sizes):
            samples, df_inds = sampler.sample(size, ignore_inds, assay, return_orig_inds=True)
            out_set += samples
            ignore_inds += df_inds
            assay = out_set[0][0]


        return out_set