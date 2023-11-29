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

		# split data, if not already done.
		if not os.path.isfile(TRAIN_SAMPLES_PATH) or not os.path.isfile(TEST_SAMPLES_PATH):

			print("Splitting data (stratified 80-20).")
			splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
			train_indices, test_indices = next(
				splitter.split(protein_data.obs, protein_data.obs['coarse.cell.type'])
			)

			train_rows = protein_data.to_df().iloc[train_indices].index.to_series(name='samples')
			train_rows.to_csv(TRAIN_SAMPLES_PATH, index=False)

			test_rows = protein_data.to_df().iloc[test_indices].index.to_series(name='samples')
			test_rows.to_csv(TEST_SAMPLES_PATH, index=False)
		
		else:
			train_rows = pd.read_csv(TRAIN_SAMPLES_PATH)['samples']
			test_rows = pd.read_csv(TEST_SAMPLES_PATH)['samples']

		## TODO: Divide based on split (stratified).
		self.protein_split = protein_data.to_df().iloc[10000:20000, :]
		self.rna_split = rna_data.to_df().iloc[10000:20000, :]

		reqd_rows = train_rows if split=='train' else test_rows
		self.protein_split = protein_data.to_df().loc[reqd_rows]
		self.rna_split = rna_data.to_df().loc[reqd_rows]


	def __len__(self):
		return self.rna_split.shape[0]
	

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x_data = np.array(self.rna_split.iloc[idx, :]).astype(np.float32)
		y_data = np.array(self.protein_split.iloc[idx, :]).astype(np.float32)

		return (x_data, y_data)