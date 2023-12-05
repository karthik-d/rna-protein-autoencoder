from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
import joblib

import pandas as pd
import torch
import os
import scanpy as sc
import numpy as np


# TODO: Don't take batch_size in dataset.


class CovidDataset(Dataset):

	def __init__(self, version='three', split='train', input_type='norm', normalization_method=None, batch_size=32):
		
		# defaults and decorations.
		self.verbose_render = False
		self.batch_size = batch_size
		
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

		# get raw layer, if reqd.
		if input_type == 'raw':
			rna_data = rna_data.raw.to_adata()
			protein_data = protein_data.raw.to_adata()

		self.split_data(rna_data, protein_data)
		self.preprocess_data()


	def __len__(self):
		return self.rna_split.shape[0]
	

	def __getitem__(self, idx):
	
		if torch.is_tensor(idx):
			idx = idx.tolist()

		x_data_df = self.rna_split.iloc[idx, :]
		y_data_df = self.protein_split.iloc[idx, :]
		
		# package into required types.
		x_data = np.array(x_data_df).astype(np.float32)
		y_data = np.array(y_data_df).astype(np.float32)

		self.batch_metadata_registered = False
		if not self.verbose_render:
			return (x_data, y_data)
		else:
			if torch.is_tensor(idx):
				return (x_data, y_data, x_data_df.columns.values, y_data_df.columns.values, x_data_df.index.values)
			else:
				return (x_data, y_data, x_data_df.index.values, y_data_df.index.values, [x_data_df.name])


	# TODO: This is torch-like, but parameterize with custom dataloader.
	def set_verbose_render(self, state):
		self.verbose_render = state


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
	

	def preprocess_data(self):
		
		# 1. Protein data normalization.
		if self.normalization_method == 'minmax':
			scaler_path = os.path.join(self.DATA_PATH, "minmax_scaler.save")
			
			if self.split != 'train':
				# read previous norm params; else, throw error.
				scaler = joblib.load(scaler_path) 
				self.protein_split = pd.DataFrame(scaler.transform(self.protein_split), columns=self.protein_split.columns)
			else:
				scaler = preprocessing.MinMaxScaler()
				self.protein_split = pd.DataFrame(scaler.fit_transform(self.protein_split), columns=self.protein_split.columns)
				joblib.dump(scaler, scaler_path)


	def collation_register_metadata(self, batch_data):
		# middleware function to interject DataLoader behavior and extract metadata.
		# TODO: define custom dataloader.

		if not self.verbose_render:
			return batch_data 
		else:
			# fill remaining samples for batch
			self.batch_metadata_registered = True
			# register metadata.
			self.curr_batch_rna_cols = batch_data[0][2]
			self.curr_batch_protein_cols = batch_data[0][3]
			self.curr_batch_cells = [batch_elem[4][0] for batch_elem in batch_data]
			# retain only reqd. data.
			batch_x_data_vec = np.array([batch_elem[0] for batch_elem in batch_data])
			batch_y_data_vec = np.array([batch_elem[1] for batch_elem in batch_data])
			# check if batch has lesser samples.
			if batch_x_data_vec.shape[0]%self.batch_size == 0:
				batch_x_data = torch.from_numpy(batch_x_data_vec.reshape((self.batch_size, -1))).to(torch.float32)
				batch_y_data = torch.from_numpy(batch_y_data_vec.reshape((self.batch_size, -1))).to(torch.float32)
			else:
				# pad with initial rows.
				n_reqd_rows = self.batch_size - batch_x_data_vec.shape[0]
				batch_x_data = torch.from_numpy(np.concatenate(
					[batch_x_data_vec, self.rna_split.iloc[range(n_reqd_rows), :].to_numpy()]
				).reshape((self.batch_size, -1))).to(torch.float32)
				batch_y_data = torch.from_numpy(np.concatenate(
					[batch_y_data_vec, self.protein_split.iloc[range(n_reqd_rows), :].to_numpy()]
				).reshape((self.batch_size, -1))).to(torch.float32)

			return (
				batch_x_data,
				batch_y_data
			)
		

	def get_curr_batch_metadata(self):
		if not self.batch_metadata_registered:
			print("Warning: Metadata was never registered for this batch; unexpected behavior.")
		return self.curr_batch_rna_cols, self.curr_batch_protein_cols, self.curr_batch_cells


		
