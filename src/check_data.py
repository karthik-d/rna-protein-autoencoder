import os
import scanpy as sc


DATA_PATH = "../data"

CITE_DATA_PATH = os.path.join(DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_CITE.h5ad")
RNA_DATA_PATH = os.path.join(DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_RNA.h5ad")


cite_df = sc.read_h5ad(CITE_DATA_PATH)
print(cite_df.n_obs, cite_df.n_vars)
print(cite_df.obs_names)
print(cite_df.to_df())
# print(cite_df.head())

rna_df = sc.read_h5ad(RNA_DATA_PATH)
print(rna_df.n_obs, rna_df.n_vars)
print(rna_df.obs_names)
print(rna_df.to_df())
# print(rna_df.head())