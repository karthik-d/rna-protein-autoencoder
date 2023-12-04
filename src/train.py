import torch
import pathlib
import os
import numpy as np

from nn.simple_autoencoder import AutoEncoder
from data.covid_dataset import CovidDataset
from utils.metrics import compute_colwise_correlations


BATCH_SIZE = 256 # change to 256, originally 32
RUN_NAME = 1


### Model Initialization

# init GPU

FLAG = torch.cuda.is_available()
device = torch.device("cuda:0" if FLAG else "cpu")
print("GPU Access:", FLAG)
print("DEVICE:", device)
if FLAG:
    print("DEVICE Name:", torch.cuda.get_device_name(0))

epoch_model_path = "../data/models/run_"+f'{RUN_NAME}'+"/epoch-{epoch}_mse-{mse:.3f}_loss-{loss:.3f}.pth"
pathlib.Path(os.path.dirname(epoch_model_path)).mkdir(
	parents = True,
	exist_ok = False
)

model = AutoEncoder().to(device)
 
# Validation using MSE Loss function

loss_function = torch.nn.CrossEntropyLoss()
mse_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer.

optimizer = torch.optim.Adam(
	model.parameters(),
    lr = 1e-3,
    weight_decay = 1e-5
)

# Specify data loader

def get_data_loaders():
	
	train_dataset = CovidDataset(split='train')
	valid_dataset = CovidDataset(split='valid')
	# wrap dataset into dataloader.
	return torch.utils.data.DataLoader(
		train_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True
	), torch.utils.data.DataLoader(
		valid_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False
	)

### TRAINING ### 

num_epochs = 50
output =[]
train_loader, valid_loader = get_data_loaders()
for epoch in range(num_epochs):
	batch_size = BATCH_SIZE

	# TRAINING PHASE --------
	model.train(mode=True)

	# Init accumulators
	train_running_loss = 0.0
	train_running_mse = 0
	train_running_pr = 0
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
			train_loss = loss_function(
				input=train_outputs,
				target=train_labels
			)
			train_mse = mse_function(train_outputs, train_labels)
			train_pr = torch.corrcoef(torch.cat([train_outputs, train_labels]))
			train_loss.backward()
			optimizer.step()

		# Update training stats
		train_running_loss += train_loss.item() * train_inputs.size(0)
		train_running_mse += train_mse.item()
		train_labels_batchwise.append(train_labels.detach().numpy())
		train_predictions_batchwise.append(train_outputs.detach().numpy())
		print(f"Training Step: {idx} of {num_train_steps}", end='\r')


	# Store training stats
	train_loss_stat = train_running_loss / len(train_loader.dataset)
	train_mse_stat = train_running_mse / len(train_loader.dataset)

	
	# CUDA cleanup
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	all_train_labels = np.concatenate(train_labels_batchwise)
	all_train_predictions = np.concatenate(train_predictions_batchwise)
	train_corr_stat = compute_colwise_correlations(all_train_labels, all_train_predictions)

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
			valid_loss = loss_function(
				input=valid_outputs, 
				target=valid_labels
			)
			valid_mse = mse_function(valid_outputs, valid_labels)
			valid_pr = torch.corrcoef(torch.cat([valid_outputs, valid_labels]))

		# Update validation stats
		valid_running_loss += valid_loss.item() * valid_inputs.size(0)
		valid_running_mse += valid_mse.item()
		valid_labels_batchwise.append(valid_labels.detach().numpy())
		valid_predictions_batchwise.append(valid_outputs.detach().numpy())


	# Store validation stats
	valid_loss_stat = valid_running_loss / len(valid_loader.dataset)
	valid_mse_stat = valid_running_mse / len(valid_loader.dataset)

	# CUDA cleanup
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		
	all_valid_labels = np.concatenate(valid_labels_batchwise)
	all_valid_predictions = np.concatenate(valid_predictions_batchwise)
	valid_corr_stat = compute_colwise_correlations(all_valid_labels, all_valid_predictions)

	# compute metrics.
	print(f'Epoch [{epoch + 1}/{num_epochs}]')
	print(f'[Training]. Loss: {train_loss_stat}, MSE: {train_mse_stat}, Corr: {np.mean(train_corr_stat)}.')
	print(f'[Validation]. Loss: {valid_loss_stat}, MSE: {valid_mse_stat}, Corr: {np.mean(valid_corr_stat)}.')
	
	# save current model --> replaced at each epoch.
	print("saving model state ...")
	torch.save(
		model.state_dict(), 
		epoch_model_path.format(epoch=epoch, loss=valid_loss_stat, mse=valid_mse_stat)
	)
    
