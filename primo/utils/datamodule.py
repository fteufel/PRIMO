import torch
import lightning as L
import pandas as pd
import hydra
import omegaconf
from tqdm import tqdm
import numpy as np
from typing import Tuple, List

from .dataset import SequenceSetDataset, TestDataset, SupportDataframeTestDataset, BaselineSequenceDataset
from .set_samplers import SimpleSetSampler, CombinedProbabilitySampler, CombinedSetSampler


class SetDataModule(L.LightningDataModule):
    def __init__(self, cfg, condition_test_on_train_data: bool = False):
        super().__init__()
        self.cfg = cfg
        self.condition_test_on_train_data = condition_test_on_train_data

    def setup(self, stage=None):

        # load all data
        df = pd.read_pickle(self.cfg.data.all_data_df)

        df = df.loc[df['mutated_sequence'].str.len() <= self.cfg.data.max_protein_length]

        df = self._prepare_transformed_labels(df)
        


        wild_type_df = pd.read_csv(self.cfg.data.wild_type_csv)

        # parse auxiliary labels and prepare dictionary to be used by the datasets
        aux_labels = {}
        for i, aux_label_file in enumerate(self.cfg.data.auxiliary_labels):
            df_aux = pd.read_csv(aux_label_file)
            # join this to the main dataframe as `auxiliary_label_i`
            df_aux = df_aux.rename(columns={'label': f'auxiliary_label_{i}'})
            df = df.merge(df_aux[['assay', 'mutated_sequence', f'auxiliary_label_{i}']],
                          on=['assay', 'mutated_sequence'], how='left')

        for i, aux_label_file in enumerate(self.cfg.data.auxiliary_labels_wild_type):
            df_aux = pd.read_csv(aux_label_file)
            # join this to the main dataframe as `auxiliary_label_i`
            df_aux = df_aux.rename(columns={'label': f'auxiliary_label_{i}_wt'})
            wild_type_df = wild_type_df.merge(df_aux[['assay', 'sequence', f'auxiliary_label_{i}_wt']],
                          on=['assay', 'sequence'], how='left')

        # parse "mutant" - we just need two positions.
        df['mutant_range_start'] = None
        df['mutant_range_end'] = None
        df['mutant_positions'] = None
        for idx, row in df.iterrows():
            # example format : N137D:A141S:V142F:V143A:E144N:C157T:F163V
            # print(row)
            mutants = row['mutant'].split(':')
            positions = []
            for mutant in mutants:
                # example format : N137D
                position = int(mutant[1:-1]) - 1 # make zero-based indexing
                positions.append(position)

            # one-based indexing
            df.at[idx, 'mutant_range_start'] = min(positions)
            df.at[idx, 'mutant_range_end'] = max(positions)
            df.at[idx, 'mutant_positions'] = positions

        # make hold out df
        if self.cfg.data.hold_out.ignore is not None:
            df = df.loc[~df['assay'].isin(self.cfg.data.hold_out.ignore)]

        if self.cfg.data.test_mode == 'hold_out':
            df_holdout = df.loc[df['assay'].isin(self.cfg.data.hold_out.test)]
            df = df.loc[~df['assay'].isin(self.cfg.data.hold_out.test)]
            print(f'Holding out {len(self.cfg.data.hold_out.test)} assays for testing.')

            # alternative we can specify a list of assays explicitly to train on
            # if we don't specify this, we train on all that were not held out in ignore or test
            if hasattr(self.cfg.data.hold_out, 'train') and self.cfg.data.hold_out.train is not None:
                df = df.loc[df['assay'].isin(self.cfg.data.hold_out.train) | df['assay'].isin(self.cfg.data.hold_out.valid)]

            # split data
            df_train, df_val, df_test = self._split(df, self.cfg.data.cv_mode)
    

        elif self.cfg.data.test_mode == 'keep_few_shot':
            # from each assay in self.cfg.data.hold_out_test, sample N observations based on stratified auxiliary_label_0 (or _1 if _0 has NaNs)
            # keep the N for training, the remainder is df_holdout 
            df_holdout = pd.DataFrame()
            df_train_fewshot = pd.DataFrame()

            for assay in self.cfg.data.hold_out.test:
                df_assay = df.loc[df['assay'] == assay].copy()

                # Determine the label to use for stratification
                if df_assay['auxiliary_label_0'].isna().any():
                    stratify_label = 'auxiliary_label_1'
                else:
                    stratify_label = 'auxiliary_label_0'

                # Sort the dataframe by the stratify label
                df_assay = df_assay.sort_values(by=stratify_label)

                # Sample N approximately equidistant points
                total_samples = 64  # Total number of samples to keep
                step = max(1, len(df_assay) // total_samples)
                df_assay_train = df_assay.iloc[::step].head(total_samples)
                df_train_fewshot = pd.concat([df_train_fewshot, df_assay_train])


                # The remainder is held out for testing
                df_assay_holdout = df_assay.drop(df_assay_train.index)
                df_holdout = pd.concat([df_holdout, df_assay_holdout])


            # Remove the holdout assays from the main dataframe
            df = df.loc[~df['assay'].isin(self.cfg.data.hold_out.test)]

            # subset to train
            if hasattr(self.cfg.data.hold_out, 'train') and self.cfg.data.hold_out.train is not None:
                if self.cfg.data.hold_out.valid is not None:
                    df_train_regular = df.loc[df['assay'].isin(self.cfg.data.hold_out.train) | df['assay'].isin(self.cfg.data.hold_out.valid)]
                else:
                    df_train_regular = df.loc[df['assay'].isin(self.cfg.data.hold_out.train)]
            else:
                df_train_regular = df

            # only split the non-holdout datasets for val
            if self.cfg.data.cv_mode != 'precomputed_random_valonly':
                raise NotImplementedError('Only precomputed_random_valonly is supported for keep_few_shot mode, the others are not tested for compatibility.')
            df_train, df_val, df_test = self._split(df_train_regular, self.cfg.data.cv_mode)

            # Add the few-shot training data to the training set
            df_train = pd.concat([df_train_fewshot, df_train_regular])

            print(f'Keeping few-shot samples from {len(self.cfg.data.hold_out.test)} assays for training.')



        # # filter out assays with fewer rows than the set size
        assay_counts = df_train['assay'].value_counts()
        df_train = df_train[df_train['assay'].isin(assay_counts[assay_counts > self.cfg.data.set_size].index)]
        assay_counts = df_val['assay'].value_counts()
        df_val = df_val[df_val['assay'].isin(assay_counts[assay_counts > self.cfg.data.set_size].index)]

        print(f'Dropped {len(assay_counts) - len(df_train["assay"].unique())} assays from training set due to insufficient data.')
        print(f'Dropped {len(assay_counts) - len(df_val["assay"].unique())} assays from validation set due to insufficient data.')


        set_sampler = self._instantiate_sampler(self.cfg, df_train, counts=self.cfg.training.sampler_counts if hasattr(self.cfg.training, 'sampler_counts') else None)
        masker = hydra.utils.instantiate(self.cfg.training.masker)


        self.train_dataset = SequenceSetDataset(
            self.cfg.data.num_labels,
            self.cfg.data.sets_per_epoch,
            self.cfg.data.set_size,
            df_train,
            set_sampler = set_sampler,
            masker = masker,
            min_pad_length=self.cfg.data.window_size,
            is_deterministic=False,
            only_mask_last=self.cfg.training.only_mask_last,
        )
        
        set_sampler = self._instantiate_sampler(self.cfg, df_val, counts=self.cfg.training.sampler_counts if hasattr(self.cfg.training, 'sampler_counts') else None)
        self.val_set_dataset = SequenceSetDataset(
            self.cfg.data.num_labels,
            self.cfg.data.sets_per_validation,
            self.cfg.data.set_size,
            df_val,
            set_sampler=set_sampler,
            masker=masker,
            min_pad_length=self.cfg.data.window_size,
            is_deterministic=self.cfg.training.deterministic_val_sets,
            only_mask_last=self.cfg.training.only_mask_last,
        )


        # test: make 1 dataset per assay
        # here we assume we do full in-context testing,
        # so we just assign set_size-1 random test points as the (labeled) support set.
        # the dataset class takes care of resampling the support set when we need to get predictions
        # for all points in the assay.
        self.test_datasets = {}
        for assay, assay_df in df_holdout.groupby('assay'):

            # if we have an assay in both train and test, we condition on the train data if specified
            if self.condition_test_on_train_data and assay in df_train['assay'].unique():
                print(f'Conditioning test data for assay {assay} on training data.')
                conditioning_df = df_train.loc[df_train['assay'] == assay].sample(self.cfg.data.set_size - 1)

                self.test_datasets[assay] = SupportDataframeTestDataset(
                    self.cfg.data.num_labels,
                    assay_df, 
                    masker=masker,
                    conditioning_dataframe = conditioning_df,
                    min_pad_length=self.cfg.data.window_size,
                )

            else:
            
                # draw set_size-1 random inds from assay_df
                conditioning_sample_inds = np.random.randint(0, len(assay_df), self.cfg.data.set_size - 1).tolist()

                # or draw stratified
                if {'_target_': 'primo.utils.set_samplers.QuantileStratifiedSetSampler'} in self.cfg.training.set_sampler:

                    assay_df['quantile'] = pd.qcut(assay_df['DMS_score_modeling'], self.cfg.data.set_size-1, labels=False, duplicates='drop')
                    # draw 1 from each quantile
                    if assay_df['quantile'].nunique() < self.cfg.data.set_size-1:
                        # draw random instead
                        conditioning_sample_inds = np.random.randint(0, len(assay_df), self.cfg.data.set_size - 1).tolist()
                        print(f'Not enough quantiles for assay {assay}, drawing random samples instead.')
                    else:
                        conditioning_sample_inds = []
                        for i in range(self.cfg.data.set_size-1):
                            sample_index = assay_df[assay_df['quantile'] == i].sample(1).index[0]
                            iloc_index = assay_df.index.get_loc(sample_index)
                            conditioning_sample_inds.append(iloc_index)


                self.test_datasets[assay] = TestDataset(
                    self.cfg.data.num_labels,
                    assay_df, 
                    masker=masker,
                    conditioning_sample_inds = conditioning_sample_inds,
                    min_pad_length=self.cfg.data.window_size,
                )

        if df_test is not None:
            for assay, assay_df in df_test.groupby('assay'):

                conditioning_sample_inds = np.random.randint(0, len(assay_df), self.cfg.data.set_size - 1).tolist()
                self.test_datasets[assay] = TestDataset(
                    self.cfg.data.num_labels,
                    assay_df, 
                    masker=masker,
                    conditioning_sample_inds=conditioning_sample_inds,
                    min_pad_length=self.cfg.data.window_size,
                )


    def _prepare_transformed_labels(self, df):
        """
        Transform the raw DMS label into the format used by the model.
        If some assays are incompatible with the transform, they are dropped.
        (e.g. if there is only one unique DMS score in the assay)
        """

        if self.cfg.data.transform == 'decile':
            # make decile scores for each assay
            df['DMS_score_decile'] = None
            for assay, assay_df in df.groupby('assay'):

                if assay_df['DMS_score'].nunique() ==1:
                    df = df.drop(df[df['assay'] == assay].index)
                    continue

                # duplicates=drop --> if 10 deciles don't exist, we have fewer quantiles.
                df.loc[df['assay'] == assay, 'DMS_score_decile'] = pd.qcut(assay_df['DMS_score'], 10, labels=False, duplicates='drop')

            # better decile score training: 0 mean, divide by 10.
            df['DMS_score_decile'] = (df['DMS_score_decile'] - 4.5) / 4.5
            df['DMS_score_modeling'] = df['DMS_score_decile']

        elif self.cfg.data.transform == 'minmax':
            # minmax scale the DMS scores
            df['DMS_score_minmax'] = None
            for assay, assay_df in df.groupby('assay'):

                if assay_df['DMS_score'].nunique() ==1:
                    df = df.drop(df[df['assay'] == assay].index)
                    continue

                min_score = assay_df['DMS_score'].min()
                max_score = assay_df['DMS_score'].max()
                df.loc[df['assay'] == assay, 'DMS_score_minmax'] = (assay_df['DMS_score'] - min_score) / (max_score - min_score + 1e-6)

            df['DMS_score_modeling'] = df['DMS_score_minmax'] - 0.5
        
        elif self.cfg.data.transform == 'batch_minmax':
            # this transform is handled in the collate_fn of the dataloader.
            df['DMS_score_modeling'] = df['DMS_score']
        elif self.cfg.data.transform == 'standard_scale':
            df['DMS_score_scaled'] = None
            for assay, assay_df in df.groupby('assay'):
                df.loc[df['assay'] == assay, 'DMS_score_scaled'] = (assay_df['DMS_score'] - assay_df['DMS_score'].mean()) / assay_df['DMS_score'].std()
            df['DMS_score_modeling'] = df['DMS_score_scaled']
        
        else:
            raise NotImplementedError(f"Transform {self.cfg.data.transform} not implemented.")

        return df

    def _instantiate_sampler(self, cfg, dataframe, weights = None, counts = None):
        """
        If the provided sampler in the cfg is a list, we need to set up a
        CombinedProbabilitySampler. Otherwise, we instantiate the sampler 
        directly. Nested instantiation is not supported by hydra.
        """
        if isinstance(cfg.training.set_sampler, omegaconf.ListConfig):
            samplers = []
            for sampler_cfg in cfg.training.set_sampler:
                sampler = hydra.utils.instantiate(sampler_cfg, dataframe=dataframe, window_size=self.cfg.data.window_size, add_special_tokens=True, return_variant_positions=True, return_aux_labels=True)
                samplers.append(sampler)

            if counts is not None:
                return CombinedSetSampler(samplers, sizes=counts)

            elif weights is not None:
                return CombinedProbabilitySampler(samplers, weights=weights)
                
        else:
            return hydra.utils.instantiate(cfg.training.set_sampler, dataframe=dataframe, window_size=self.cfg.data.window_size, add_special_tokens=True, return_variant_positions=True, return_aux_labels=True)
            


    def _split(self, df, mode='random'):
        """
        Call this after removing the ``test`` and ``ignore`` assays from the dataframe.
        """

        if mode == 'random':
            df = df.sample(frac=1)
            train_size = int(0.8 * len(df))
            val_size = int(0.05 * len(df))
            train_df = df[:train_size]
            val_df = df[train_size:train_size + val_size]
            test_df = df[train_size + val_size:]
        elif mode == 'precomputed_random':
            train_df = df.loc[df['fold_random_5'] != 0]
            test_df = df.loc[df['fold_random_5'] == 0]

            val_size = int(0.05 * len(train_df))
            train_df = train_df.sample(frac=1)
            val_df = train_df[:val_size]
            train_df = train_df[val_size:]
        elif mode == 'hold_out_assay':
            train_df = df.loc[~df['assay'].isin(self.cfg.data.hold_out.valid)]
            val_df = df.loc[df['assay'].isin(self.cfg.data.hold_out.valid)]
            test_df = None
        elif mode == 'precomputed_random_valonly':
            val_size = int(0.05 * len(df))
            val_df = df.sample(val_size)
            train_df = df.loc[~df.index.isin(val_df.index)]
            test_df = None
            
        else:
            raise NotImplementedError
        
        return train_df, val_df, test_df



    def train_dataloader(self):
        # return a dataloader
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.cfg.training.batch_size, num_workers=self.cfg.training.num_workers, collate_fn=self.train_dataset.minmax_scale_collate_fn if self.cfg.data.transform == 'batch_minmax' else self.train_dataset.collate_fn
        )

    def val_dataloader(self):
        dl = {'sets': torch.utils.data.DataLoader(
            self.val_set_dataset, batch_size=self.cfg.training.batch_size, num_workers=self.cfg.training.val_num_workers, collate_fn= self.val_set_dataset.minmax_scale_collate_fn if self.cfg.data.transform == 'batch_minmax' else self.val_set_dataset.collate_fn
        )
                }
        return dl

    def test_dataloader(self):
        
        dl = {
            assay: torch.utils.data.DataLoader(
                dataset, batch_size=self.cfg.training.batch_size, num_workers=self.cfg.training.val_num_workers, collate_fn=dataset.minmax_scale_collate_fn if self.cfg.data.transform == 'batch_minmax' else dataset.collate_fn
            )
            for assay, dataset in self.test_datasets.items()
        }
        return dl
    
    @staticmethod
    def _few_shot_selection(df_assay, n, stratification_mode='zeroshot', **kwargs):
        """
        Select n samples from each assay based on stratified sampling.
        """

        if stratification_mode == 'zeroshot':
            # Sort the dataframe by the stratify label
            df_assay = df_assay.sort_values(by=kwargs.get('stratify_label', 'auxiliary_label_0'))

            # Sample N approximately equidistant points
            total_samples = n
            step = max(1, len(df_assay) // total_samples)
            df_assay_train = df_assay.iloc[::step].head(total_samples)
        elif stratification_mode == 'random':
            df_assay_train = df_assay.sample(n, seed=kwargs.get('seed', 123))
        elif stratification_mode == 'random_increasing':
            # ensure that if we first do e.g. 16, then 32, 64 etc, 32 includes the first 16, and so on.
            # we do this by sampling one by one, and not allowing duplicates.
            rng = np.random.default_rng(seed=kwargs.get('seed', 123))
            sample_inds = []
            for i in range(n):
                new_sample_ind = rng.choice(df_assay.index.difference(sample_inds), 1, replace=False)
                sample_inds.extend(new_sample_ind)

        df_assay_train = df_assay.loc[sample_inds]



        return df_assay_train
    

    def get_finetuning_dataloaders(
            self, 
            assay, 
            n, 
            test_mode: str = 'all', 
            data_selection_mode='random_increasing',
            normalization: str = None,
            seed: int = 123,
            ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Get a dataloader for finetuning on the assay, test on the remaining data from the assay.

        We also use the finetuning data as the support set when conditioning.
        """

        if test_mode  not in  ['all', 'remainder']:
            raise ValueError(f"test_mode {test_mode} not supported.")

        df_assay = self.test_datasets[assay].assay_dataframe_test

        
        # Determine the label to use for stratification
        if df_assay['auxiliary_label_0'].isna().any():
            stratify_label = 'auxiliary_label_1'
        else:
            stratify_label = 'auxiliary_label_0'

        # df_assay_train = self._few_shot_selection(df_assay, n, stratification_mode='zeroshot', stratify_label=stratify_label)
        df_assay_train = self._few_shot_selection(df_assay, n, stratification_mode=data_selection_mode, stratify_label=stratify_label, seed=seed)

        # The remainder is held out for testing
        df_assay_holdout = df_assay.drop(df_assay_train.index) if test_mode == 'remainder' else df_assay

        if normalization == 'minmax':
            min_score = df_assay_train['DMS_score'].min()
            max_score = df_assay_train['DMS_score'].max()
            # if we have N=1, make the reference point the middle of the range
            df_assay_train['DMS_score_modeling'] = (df_assay_train['DMS_score'] - min_score) / (max_score - min_score + 1e-6) if len(df_assay_train) > 1 else 0.5
            df_assay_holdout['DMS_score_modeling'] = (df_assay_holdout['DMS_score'] - min_score) / (max_score - min_score + 1e-6)

        train_dataset = SequenceSetDataset(
            self.cfg.data.num_labels,
            self.cfg.data.sets_per_epoch,
            self.cfg.data.set_size,
            df_assay_train,
            set_sampler = self._instantiate_sampler(self.cfg, df_assay_train, counts=self.cfg.training.sampler_counts if hasattr(self.cfg.training, 'sampler_counts') else None),
            masker = hydra.utils.instantiate(self.cfg.training.masker),
            min_pad_length=self.cfg.data.window_size,
            is_deterministic=False,
            only_mask_last=self.cfg.training.only_mask_last,
        )

        set_size_test = self.cfg.data.set_size_test if 'set_size_test' in self.cfg.data else self.cfg.data.set_size

        # condition_sample_inds = np.random.randint(0, len(df_assay_holdout), self.cfg.data.set_size - 1).tolist()
        condition_sample_inds = np.random.choice(len(df_assay_train), set_size_test - 1, replace=False).tolist()
        test_dataset = TestDataset(
            self.cfg.data.num_labels,
            df_assay_holdout,
            masker=hydra.utils.instantiate(self.cfg.training.masker),
            conditioning_sample_inds = condition_sample_inds,
            assay_dataframe_support=df_assay_train,
            min_pad_length=None,
        )

        return torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg.training.batch_size, num_workers=self.cfg.training.num_workers, collate_fn=train_dataset.collate_fn
        ), torch.utils.data.DataLoader(
            test_dataset, batch_size=self.cfg.training.batch_size, num_workers=self.cfg.training.num_workers, collate_fn=test_dataset.collate_fn
        )
    

    def get_baseline_dataloaders(
            self, 
            assay, 
            n, 
            test_mode: str = 'all', 
            indel_mode: str = 'include', 
            data_selection_mode='random_increasing',
            normalization: str = 'z_score',
            seed: int = 123,
            ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            """
            Get a dataloader for training on a single assay

            indel_mode: 'include' for both, 'exclude' for snvs only, 'only' for indel only
            """

            if test_mode  not in  ['all', 'remainder']:
                raise ValueError(f"test_mode {test_mode} not supported.")

            if indel_mode not in ['include', 'exclude', 'both']:
                raise ValueError(f"indel_mode {indel_mode} not supported.")

            df_assay = self.test_datasets[assay].assay_dataframe_test

            # Determine the label to use for stratification
            if df_assay['auxiliary_label_0'].isna().any():
                stratify_label = 'auxiliary_label_1'
            else:
                stratify_label = 'auxiliary_label_0'


            if indel_mode == 'include':
                pass
            elif indel_mode == 'exclude':
                df_assay = df_assay.loc[~df_assay['file'].str.endswith('indels.csv')]
            elif indel_mode == 'only':
                df_assay = df_assay.loc[df_assay['file'].str.endswith('indels.csv')]
                # kill all-nan structure aux label
                df_assay =  df_assay.drop('auxiliary_label_1', axis=1)
            else:
                raise ValueError
            

            # Sort the dataframe by the stratify label
            # df_assay_train = self._few_shot_selection(df_assay, n, stratification_mode='zeroshot', stratify_label=stratify_label)
            df_assay_train = self._few_shot_selection(df_assay, n, stratification_mode=data_selection_mode, stratify_label=stratify_label, seed=seed)

            # The remainder is held out for testing
            df_assay_holdout = df_assay.drop(df_assay_train.index) if test_mode == 'remainder' else df_assay

            if normalization == 'z_score':
                mean = df_assay_train['DMS_score'].mean()
                std = df_assay_train['DMS_score'].std()
                df_assay_train['DMS_score_modeling'] = (df_assay_train['DMS_score'] - mean) / std
                df_assay_holdout['DMS_score_modeling'] = (df_assay_holdout['DMS_score'] - mean) / std
            

            train_dataset = BaselineSequenceDataset(
                self.cfg.data.num_labels,
                df_assay_train,
                min_pad_length=None,#self.cfg.data.window_size,
            )

            test_dataset = BaselineSequenceDataset(
                self.cfg.data.num_labels,
                df_assay_holdout,
                min_pad_length=None,#self.cfg.data.window_size,
            )

            return torch.utils.data.DataLoader(
                train_dataset, batch_size=self.cfg.training.batch_size, num_workers=self.cfg.training.num_workers, collate_fn=train_dataset.collate_fn
            ), torch.utils.data.DataLoader(
                test_dataset, batch_size=self.cfg.training.batch_size, num_workers=self.cfg.training.num_workers, collate_fn=test_dataset.collate_fn
            )


    def get_data_for_bayes_opt(self, assay, already_selected: List[int] = None, seed=123, normalization='z_score'):

        df_assay = self.test_datasets[assay].assay_dataframe_test

        if already_selected is None:
            already_selected = []

        df_train = df_assay.loc[df_assay.index.isin(already_selected)].copy()
        df_candidate = df_assay.loc[~df_assay.index.isin(already_selected)].copy()

        if normalization == 'z_score':
            df_train['DMS_score_modeling'] = (df_train['DMS_score'] - df_train['DMS_score'].mean()) / df_train['DMS_score'].std()
        elif normalization == 'minmax':
            min_score = df_train['DMS_score'].min()
            max_score = df_train['DMS_score'].max()
            df_train['DMS_score_modeling'] = (df_train['DMS_score'] - min_score) / (max_score - min_score + 1e-6)

        return df_train, df_candidate