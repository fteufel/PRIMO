"""
Script to reformat ProteinGym into one flat file.

Needs: 

- `DMS_ProteinGym_substitutions.zip`
- `DMS_ProteinGym_indels.zip`

Properties
 - 0: Stability
 - 1: Solubility
 - 2: Enzymatic activity
 - 3: Abundance
 - 4: Fluoresence (GFP)
 - 5: Binding

"""
import pandas as pd
import os
from Bio import Align

def find_deleted_indices(ref_seq, comp_seq):
    """This will also map point mutations as 2 indels, if the aligner feels like it."""
    aligner = Align.PairwiseAligner()
    alignments = aligner.align(ref_seq, comp_seq)

    # Get the first alignment (there may be multiple with the same score)
    alignment = alignments[0]

    # Get the aligned sequences
    aligned_ref_seq = str(alignment[0])
    aligned_comp_seq = str(alignment[1])
    

    changes = []
    for i in range(len(aligned_ref_seq)):
        if aligned_ref_seq[i] != aligned_comp_seq[i]:
            original = aligned_ref_seq[i] if aligned_ref_seq[i] != '-' else '-'
            new = aligned_comp_seq[i] if aligned_comp_seq[i] != '-' else '-'
            changes.append(f"{original}{i+1}{new}")

    return ":".join(changes)

property_dict = {
    "Stability" : 0,
    "Solubility" : 1,
    "Enzymatic Activity" : 2,
    "Expression": 3,
    "Fluorescence" : 4,
    "Binding" : 5
}

ignore_list = [
    # we ignore this one because it's non-GFP Fluorescence
    'PHOT_CHLRE_Chen_2023'
]
df_wt = pd.read_csv('data/all_wt.csv')
wt_dict = df_wt.set_index('assay')['sequence'].to_dict()


dfs = []

df_annot = pd.read_csv('data/DMS_overview.tsv', sep='\t')
df_annot = df_annot.loc[~df_annot['DMS_id'].isin(ignore_list)]
df_annot = df_annot.loc[df_annot['curated_selection_property'].isin(
    list(property_dict.keys())
    )]
df_annot['assay'] = df_annot['DMS_filename'].str.split('.').str[0]

for f in os.listdir('data/unformatted'):
    if f.endswith('.csv'):
        assay = f.split('.')[0].removesuffix('_indels')
        if assay not in df_annot['assay'].values:
            continue
        df = pd.read_csv(os.path.join('data/unformatted/', f))
        df['assay'] = assay
        df['file'] = f
        if 'indels' in f:
           df['mutant'] = ''
           for idx, row in df.iterrows():
               df.loc[idx, 'mutant'] = find_deleted_indices(wt_dict[assay], row['mutated_sequence'])
        df['property_int'] = property_dict[df_annot.loc[df_annot['assay'] == assay, 'curated_selection_property'].values[0]]
        dfs.append(df)


df = pd.concat(dfs)

# ~10 dupes in proteingym indel/regular.
df = df.loc[~df[['assay', 'mutated_sequence']].duplicated()]

# print summary stats - counts per property
print(pd.DataFrame(df['property_int'].value_counts()).sort_index())

df.to_csv('data/all_samples.csv')
df.to_pickle('data/all_samples.pkl')



# prepare WT table
# df_all = df_annot[['assay', 'sequence', 'pdb_file']]
# df_all.to_csv('data/all_wt.csv', index=False)