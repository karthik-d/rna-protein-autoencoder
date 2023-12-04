import torch
import pathlib
import os
import itertools
import numpy as np
import pandas as pd

from nn.simple_autoencoder import AutoEncoder
from data.covid_dataset import CovidDataset
from utils.metrics import compute_colwise_correlations


# PARAM SWEEP ---------------------
LEARNING_RATES = [ 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 ]
DECAY_RATES = [ 1e-2, 1e-3 ]
INPUT_TYPES = ['norm', 'raw']
LATENT_SPACES = [ 8, 16, 24 ]
# ---------------------------------



### Model Configuration

# init GPU

FLAG = torch.cuda.is_available()
device = torch.device("cuda:0" if FLAG else "cpu")
print("GPU Access:", FLAG)
print("DEVICE:", device)
if FLAG:
    print("DEVICE Name:", torch.cuda.get_device_name(0))


# configure path

run_path = "../data/models/{run_name}"
epoch_model_path = os.path.join(run_path, "epoch-{epoch}_corr-{corr:.3f}_loss-{loss:.3f}.pth")
training_summary_path = os.path.join(run_path, "training-summary.csv")

 
# Validation using MSE Loss function

mae_function = torch.nn.L1Loss()
mse_function = torch.nn.MSELoss()

# Load data 

def get_data_loaders(batch_size, input_type):
	
	# TODO: use `input_type` to determine the data input type -- norm and raw.

	train_dataset = CovidDataset(version='two', split='train', input_type=input_type)
	valid_dataset = CovidDataset(version='two', split='valid', input_type=input_type)
	# wrap dataset into dataloader.
	return torch.utils.data.DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True
	), torch.utils.data.DataLoader(
		valid_dataset,
		batch_size=batch_size,
		shuffle=False
	)


def train_job(
	learning_rate,
	decay_rate,
	latent_space,
	run_name,
	output_activation,
	train_loader,
	valid_loader,
	num_epochs = 100
):
	
	# lazy creation; create model saving path.
	pathlib.Path(run_path.format(run_name=run_name)).mkdir(
		parents = True,
		exist_ok = True
	)
	
	model = AutoEncoder(
		n_latent_space=latent_space,
		output_activation=output_activation
	).to(device)
	
	# Using an Adam Optimizer.
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr = learning_rate,
		weight_decay = decay_rate
	)

	# init accumulators.
	train_stats = dict(
		loss = [],
		mse = [],
		corr = [],
	)
	valid_stats = dict(
		loss = [],
		mse = [],
		corr = []
	)
	for epoch in range(num_epochs):

		# TRAINING PHASE --------
		model.train(mode=True)

		# Init accumulators
		train_running_loss = 0.0
		train_running_mse = 0
		train_labels_batchwise = []
		train_predictions_batchwise = []

		# Train over all training data -- batch-wise
		num_train_steps = len(train_loader)
		print()
		for idx, (inputs, labels) in enumerate(train_loader):
			train_inputs = inputs.to(device=device)
			train_labels = labels.to(device=device)
			optimizer.zero_grad()

			# Propagate forward and back
			with torch.set_grad_enabled(mode=True):
				train_outputs = model(train_inputs)
				train_loss = mae_function(
					input=train_outputs,
					target=train_labels
				)
				train_mse = mse_function(train_outputs, train_labels)
				train_loss.backward()
				optimizer.step()

			# Update training stats
			train_running_loss += train_loss.item() * train_inputs.size(0)
			train_running_mse += train_mse.item()
			train_labels_batchwise.append(train_labels.cpu().detach().numpy())
			train_predictions_batchwise.append(train_outputs.cpu().detach().numpy())
			print(f"Training Step: {idx} of {num_train_steps}", end='\r')

		# CUDA cleanup
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		all_train_labels = np.concatenate(train_labels_batchwise)
		all_train_predictions = np.concatenate(train_predictions_batchwise)
		
		# Store training stats
		train_stats['loss'].append(train_running_loss / len(train_loader.dataset))
		train_stats['mse'].append(train_running_mse / len(train_loader.dataset))
		train_stats['corr'].append(np.mean(compute_colwise_correlations(all_train_labels, all_train_predictions)))

		
		# VALIDATION PHASE --------
		model.train(mode=False)

		# Init accumulators
		valid_running_loss = 0.0
		valid_running_mse = 0
		valid_labels_batchwise = []
		valid_predictions_batchwise = []

		# Feed forward over all the validation data.
		for idx, (valid_inputs, valid_labels) in enumerate(valid_loader):
			valid_inputs = valid_inputs.to(device=device)
			valid_labels = valid_labels.to(device=device)

			# Feed-Forward ONLY!
			with torch.set_grad_enabled(mode=False):
				valid_outputs = model(valid_inputs)
				valid_loss = mae_function(
					input=valid_outputs, 
					target=valid_labels
				)
				valid_mse = mse_function(valid_outputs, valid_labels)

			# Update validation stats
			valid_running_loss += valid_loss.item() * valid_inputs.size(0)
			valid_running_mse += valid_mse.item()
			valid_labels_batchwise.append(valid_labels.cpu().detach().numpy())
			valid_predictions_batchwise.append(valid_outputs.cpu().detach().numpy())

		# CUDA cleanup
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			
		all_valid_labels = np.concatenate(valid_labels_batchwise)
		all_valid_predictions = np.concatenate(valid_predictions_batchwise)

		# Store validation stats
		valid_stats['loss'].append(valid_running_loss / len(valid_loader.dataset))
		valid_stats['mse'].append(valid_running_mse / len(valid_loader.dataset))
		valid_stats['corr'].append(np.mean(compute_colwise_correlations(all_valid_labels, all_valid_predictions)))

		# compute metrics.
		print(f'Epoch [{epoch + 1}/{num_epochs}]')
		print(f"[Training]. Loss: {train_stats['loss'][-1]}, MSE: {train_stats['mse'][-1]}, Corr: {np.mean(train_stats['corr'][-1])}.")
		print(f"[Validation]. Loss: {valid_stats['loss'][-1]}, MSE: {valid_stats['mse'][-1]}, Corr: {np.mean(valid_stats['corr'][-1])}.")
		
		# save current model --> replaced at each epoch.
		print("saving model state ...")
		torch.save(
			model.state_dict(), 
			epoch_model_path.format(
				epoch=epoch, 
				loss=valid_stats['loss'][-1], 
				corr=valid_stats['corr'][-1], 
				run_name=run_name
			)
		)

	# save training stats.
	stat_names = ['loss', 'mse', 'corr']
	pd.DataFrame({**{
		f'train_{stat}': train_stats[stat]
		for stat in stat_names
	}, **{
		f'valid_{stat}': valid_stats[stat]
		for stat in stat_names
	}}).to_csv(training_summary_path.format(run_name=run_name))


	# save training plots.





# -------------- START Training ---------------



train_loader, valid_loader = None, None
for inp_type in INPUT_TYPES:

	# load data. caching to reduce data loading calls.
	train_loader, valid_loader = get_data_loaders(
		batch_size = 256, # change to 256 (totalVI), originally 32
		input_type = inp_type,
		normalization_method = None     # can be: [None, 'minmax']
	)

	for lr, dr, n_latent_space in itertools.product(
		LEARNING_RATES, DECAY_RATES, LATENT_SPACES
	):
		run_name = f"run_lr-{lr:.2e}_dr-{dr:.2e}_ls-{n_latent_space}_inp-{inp_type}"
		train_job(
			learning_rate = lr,
			decay_rate = dr,
			latent_space = n_latent_space,
			run_name = run_name,
			output_activation='linear',     # can be: ['linear', 'sigmoid']
			train_loader = train_loader,
			valid_loader = valid_loader,
			num_epochs = 2
		)
