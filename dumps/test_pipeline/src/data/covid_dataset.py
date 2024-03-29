from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import torch
import os
import scanpy as sc
import numpy as np


class CovidDataset(Dataset):

	def __init__(self, split='train'):
		
		DATA_PATH = "../data"
		PROTEIN_DATA_PATH = os.path.join(DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_CITE.h5ad")
		RNA_DATA_PATH = os.path.join(DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_RNA.h5ad")
		
		TRAIN_SAMPLES_PATH = os.path.join(DATA_PATH, "covid-flu_train-samples.csv")
		TEST_SAMPLES_PATH = os.path.join(DATA_PATH, "covid-flu_test-samples.csv")

		print("Protein Data.")
		protein_data = sc.read_h5ad(PROTEIN_DATA_PATH)
		print(protein_data.n_obs, protein_data.n_vars)
		print(protein_data.to_df())

		print("RNA Data.")
		rna_data = sc.read_h5ad(RNA_DATA_PATH)
		print(rna_data.n_obs, rna_data.n_vars)
		print(rna_data.to_df())

		## TODO: Divide based on split (stratified).
		self.protein_split = protein_data.to_df().iloc[:10000, :]
		self.rna_split = rna_data.to_df().iloc[:10000, :]

		# reqd_rows = train_rows if split=='train' else test_rows
		# self.protein_split = protein_data.to_df().loc[reqd_rows]
		# self.rna_split = rna_data.to_df().loc[reqd_rows]


	def __len__(self):
		return self.rna_split.shape[0]
	

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x_data = np.array(self.rna_split.iloc[idx, :]).astype(np.float32)
		y_data = np.array(self.protein_split.iloc[idx, :]).astype(np.float32)

		return (x_data, y_data)