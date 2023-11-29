from torch.utils.data import Dataset
import torch
import os
import scanpy as sc
import numpy as np


class CovidDataset(Dataset):

	def __init__(self, split='train'):
		
		DATA_PATH = "../data"
		PROTEIN_DATA_PATH = os.path.join(DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_CITE.h5ad")
		RNA_DATA_PATH = os.path.join(DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_RNA.h5ad")

		print("Protein Data.")
		protein_data = sc.read_h5ad(PROTEIN_DATA_PATH)
		print(protein_data.n_obs, protein_data.n_vars)
		print(protein_data.to_df())

		print("RNA Data.")
		rna_data = sc.read_h5ad(RNA_DATA_PATH)
		print(rna_data.n_obs, rna_data.n_vars)
		print(rna_data.to_df())

		## TODO: Divide based on split (stratified).
		self.protein_split = protein_data.to_df().iloc[10000:20000, :]
		self.rna_split = rna_data.to_df().iloc[10000:20000, :]


	def __len__(self):
		return self.rna_split.shape[0]
	

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x_data = np.array(self.rna_split.iloc[idx, :]).astype(np.float32)
		y_data = np.array(self.protein_split.iloc[idx, :]).astype(np.float32)

		return (x_data, y_data)