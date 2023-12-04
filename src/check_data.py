import os
import scanpy as sc


DATA_PATH = "../data/version_three"

CITE_DATA_PATH = os.path.join(DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_CITE.h5ad")
RNA_DATA_PATH = os.path.join(DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_RNA.h5ad")


# cite_df = sc.read_h5ad(CITE_DATA_PATH)
# print(cite_df.n_obs, cite_df.n_vars)
# print(cite_df.obs)
# print(cite_df.head())

rna_df = sc.read_h5ad(RNA_DATA_PATH)
rna_df_raw = rna_df.raw.to_adata() # this creates another adata obj w/ raw matrix as initial input

print(rna_df.n_obs, rna_df.n_vars)

print('---- print normalized data ----')
print(rna_df.to_df().iloc[0:5,0:10])
print('---- print raw data ----')
print(rna_df_raw.to_df().iloc[0:5,0:10])

# print(rna_df.obs_names[0:5])
# print(rna_df.layers.keys())
# print(rna_df.head())
