from torch.utils.data import Dataset
import torch
import os
import scanpy as sc


class CovidDataset(Dataset):

	def __init__(self, split='train'):
		
		DATA_PATH = "../data"
		PROTEIN_DATA_PATH = os.path.join(DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_CITE.h5ad")
		RNA_DATA_PATH = os.path.join(DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_RNA.h5ad")

		print("Protein Data.")
		protein_data = sc.read_h5ad(PROTEIN_DATA_PATH)
		print(protein_data.n_obs, protein_data.n_vars)
		print(protein_data.obs_names)
		print(protein_data.to_df())

		print("RNA Data.")
		rna_data = sc.read_h5ad(RNA_DATA_PATH)
		print(rna_data.n_obs, rna_data.n_vars)
		print(rna_data.obs_names)
		print(rna_data.to_df())

		## TODO: Divide based on split.
		self.protein_split = protein_data.to_df()
		self.rna_split = rna_data.to_df()


	def __len__(self):
		return len(self.landmarks_frame)
	

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x_data = self.rna_split.iloc[idx, :]
		y_data = self.protein_split.iloc[idx, :]

		return (x_data, y_data)