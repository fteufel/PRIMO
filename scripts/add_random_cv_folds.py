"""
Prepare random split. Take random split from ProteinGym,
add additional random split IDs for datasets not in ProteinGym.

"""

import pandas as pd
import os
import numpy as np

dfs = []

for f in os.listdir('data/cv_folds/cv_folds_indels'):
    if f.endswith('.csv'):
        df = pd.read_csv(f'data/cv_folds/cv_folds_indels/{f}')
        df['file'] = f.split('/')[-1]
        dfs.append(df)

for f in os.listdir('data/cv_folds/substitutions_singles'):
    if f.endswith('.csv'):
        df = pd.read_csv(f'data/cv_folds/substitutions_singles/{f}')
        df['file'] = f.split('/')[-1]
        dfs.append(df)

for f in os.listdir('data/cv_folds/cv_folds_multiples_substitutions'):
    if f.endswith('.csv'):
        df = pd.read_csv(f'data/cv_folds/cv_folds_multiples_substitutions/{f}')
        df['file'] = f.split('/')[-1]
        dfs.append(df)


df_split = pd.concat(dfs).reset_index(drop=True)

# combine fold_rand_multiples and fold_random_5 - we pool indels and substitutions.
# use fold_random_5 column for that.
df_split.loc[df_split['fold_random_5'].isna(), 'fold_random_5'] = df_split.loc[df_split['fold_random_5'].isna(), 'fold_rand_multiples']
# deduplicate on file, mutated_sequence
df_split = df_split.drop_duplicates(subset=['file', 'mutated_sequence'])

df_data = pd.read_pickle('data/all_samples.pkl')

# merge on df_data file, mutated_sequence . df_split file, mutated_sequence.
# NaNs in df_split file are ok - they are the ones that are not in the split. keep and make splits ourselves.
df_data = df_data.merge(df_split[['file', 'mutated_sequence', 'fold_random_5', 'fold_modulo_5', 'fold_contiguous_5']], on=['file', 'mutated_sequence'], how='left', suffixes=('', '_split'))

rng = np.random.default_rng(42)

# prepare random split for SoluProtMutDb and FireProtDb
# for each assay, randomly split observations into 5 folds.
for assay in df_data.loc[df_data['fold_random_5'].isna()]['assay'].unique():
    df_data_assay = df_data[df_data['assay'] == assay]
    fold_random_5 = rng.integers(0, 5, df_data_assay.shape[0])
    df_data.loc[df_data['assay'] == assay, 'fold_random_5'] = fold_random_5


df_data.to_csv('data/all_samples_with_folds.csv', index=False)
df_data.to_pickle('data/all_samples_with_folds.pkl')