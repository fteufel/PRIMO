import pandas as pd
import argparse
import torch
from tqdm.auto import tqdm
import h5py
import os

from esm import pretrained
from hashlib import md5

def prepare_embeddings(df, out_path, mode='w', esm_model=None, esm_alphabet=None):

    if (esm_model is None) or (esm_alphabet is None):
        esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
    

    with torch.no_grad():
        if torch.cuda.is_available():
            esm_model = esm_model.cuda()

        batch_converter = esm_alphabet.get_batch_converter()

        with h5py.File(out_path, mode) as h5f:
            for idx, row in tqdm(df.iterrows()):

                sequence = row['mutated_sequence']
                labels, strs, toks = batch_converter([("seq", sequence)])
                if torch.cuda.is_available():
                    toks = toks.to(device="cuda", non_blocking=True)

                output = esm_model(toks, repr_layers=[33], return_contacts=False)
                embeddings = output["representations"][33][:]
                # Store data as HDF5
                h5f.create_dataset(md5(sequence.encode()).digest().hex(), data=embeddings.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("input_wt", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    esm_model, esm_alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')

    for assay, assay_df in df.groupby('assay'):

        if assay == 'HIS7_YEAST_Pokusaeva_2019':
            continue
        cleaned_assay_name = assay.replace('/', '_').replace('.', '_')
        # skip if file already exists
        if os.path.exists(f"{args.output_dir}/{cleaned_assay_name}.hdf5"):
            continue
        prepare_embeddings(assay_df, f"{args.output_dir}/{cleaned_assay_name}.hdf5", mode='w', esm_model=esm_model, esm_alphabet=esm_alphabet)


    # add wt embeddings
    # just make it work using the loop we already have
    wt_df = pd.read_csv(args.input_wt)
    wt_df['mutated_sequence'] = wt_df['sequence']
    for assay, assay_df in wt_df.groupby('assay'):
        cleaned_assay_name = assay.replace('/', '_').replace('.', '_')
        # we only do a single sequence each here, just try-except in case we need to restart this
        try:
            prepare_embeddings(assay_df, f"{args.output_dir}/{cleaned_assay_name}.hdf5", mode='a', esm_model=esm_model, esm_alphabet=esm_alphabet)
        except:
            pass
