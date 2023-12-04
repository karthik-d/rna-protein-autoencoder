from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.externals import joblib

import pandas as pd
import torch
import os
import scanpy as sc
import numpy as np



class CovidDataset(Dataset):

	def __init__(self, version='three', split='train', input_type='norm', normalization_method=None):
		
		self.DATA_PATH = os.path.join("../data", f"version_{version}")
		self.PROTEIN_DATA_PATH = os.path.join(self.DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_CITE.h5ad")
		self.RNA_DATA_PATH = os.path.join(self.DATA_PATH, "covid-flu_HC_D0_selectedCellTypes_RNA.h5ad")
		
		self.TRAIN_SAMPLES_PATH = os.path.join(self.DATA_PATH, "covid-flu_train-samples.csv")
		self.VALID_SAMPLES_PATH = os.path.join(self.DATA_PATH, "covid-flu_valid-samples.csv")
		self.TEST_SAMPLES_PATH = os.path.join(self.DATA_PATH, "covid-flu_test-samples.csv")

		self.input_type = input_type
		self.split = split
		self.normalization_method = normalization_method
		
		print("\nRNA Data.")
		rna_data = sc.read_h5ad(self.RNA_DATA_PATH)
		print(rna_data.n_obs, rna_data.n_vars)
		
		print("\nProtein Data.")
		protein_data = sc.read_h5ad(self.PROTEIN_DATA_PATH)
		print(protein_data.n_obs, protein_data.n_vars)

		self.split_data(rna_data, protein_data)
		self.preprocess_data()


	def __len__(self):
		return self.rna_split.shape[0]
	

	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x_data = np.array(self.rna_split.iloc[idx, :]).astype(np.float32)
		y_data = np.array(self.protein_split.iloc[idx, :]).astype(np.float32)

		return (x_data, y_data)
	

	def get_input_type(self):
		return self.input_type
	

	def split_data(self, rna_data, protein_data):
		
		# split data, if not already done.
		if not os.path.isfile(self.TRAIN_SAMPLES_PATH
						) or not os.path.isfile(self.TEST_SAMPLES_PATH
							  ) or not os.path.isfile(self.VALID_SAMPLES_PATH):

			print("Splitting data (stratified 80-20) ...")
			splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
			train_indices, test_indices = next(
				splitter.split(protein_data.obs, protein_data.obs['coarse.cell.type'])
			)

			# split train data again: 80-20.
			train_indices, valid_indices = next(
				splitter.split(
					protein_data.obs.iloc[train_indices], 
					protein_data.obs.iloc[train_indices]['coarse.cell.type']
			))

			train_rows = protein_data.to_df().iloc[train_indices].index.to_series(name='samples')
			train_rows.to_csv(self.TRAIN_SAMPLES_PATH, index=False)

			valid_rows = protein_data.to_df().iloc[valid_indices].index.to_series(name='samples')
			valid_rows.to_csv(self.VALID_SAMPLES_PATH, index=False)

			test_rows = protein_data.to_df().iloc[test_indices].index.to_series(name='samples')
			test_rows.to_csv(self.TEST_SAMPLES_PATH, index=False)
		
		else:
			train_rows = pd.read_csv(self.TRAIN_SAMPLES_PATH)['samples']
			valid_rows = pd.read_csv(self.VALID_SAMPLES_PATH)['samples']
			test_rows = pd.read_csv(self.TEST_SAMPLES_PATH)['samples']

		reqd_rows = {
			'train': train_rows,
			'valid': valid_rows,
			'test': test_rows
		}[self.split]

		self.protein_split = protein_data.to_df().loc[reqd_rows]
		self.rna_split = rna_data.to_df().loc[reqd_rows]
	

	def preprocess_input(self):
		
		# 1. Data normalization.
		if self.normalization_method == 'minmax':
			scaler_path = os.path.join(self.DATA_PATH, "minmax_scaler.save")
			
			if self.split != 'train':
				# read previous norm params; else, throw error.
				scaler = joblib.load(scaler_path) 
				self.rna_split = pd.DataFrame(scaler.transform(self.rna_split), columns=self.rna_split.columns)
			else:
				scaler = preprocessing.MinMaxScaler()
				self.rna_split = pd.DataFrame(scaler.fit_transform(self.rna_split), columns=self.rna_split.columns)
				joblib.dump(scaler, scaler_path)
