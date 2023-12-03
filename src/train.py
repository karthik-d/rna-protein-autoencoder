import torch
import pathlib
import time
import os

from nn.simple_autoencoder import AutoEncoder
from data.covid_dataset import CovidDataset


BATCH_SIZE = 32
RUN_NAME = 1


print("GPU Access:", torch.cuda.is_available())
epoch_model_path = "../data/models/run_"+f'{RUN_NAME}'+"/epoch-{epoch}_mse-{mse:.3f}_loss-{loss:.3f}.pth"
pathlib.Path(os.path.dirname(epoch_model_path)).mkdir(
	parents = True,
	exist_ok = False
)

# Model Initialization
device = 'cpu'
# device = 'cuda:0'
model = AutoEncoder().to(device)
 
# Validation using MSE Loss function
loss_function = torch.nn.CrossEntropyLoss()
mse_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer.
optimizer = torch.optim.Adam(
	model.parameters(),
    lr = 1e-3,
    weight_decay = 1e-3
)


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


num_epochs = 50
output =[]
train_loader, valid_loader = get_data_loaders()
for epoch in range(num_epochs):
	batch_size = BATCH_SIZE
	epoch_start_time = time.time()

	# TRAINING PHASE --------
	model.train(mode=True)

	# Init accumulators
	train_running_loss = 0.0
	train_running_mse = 0

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
			train_loss.backward()
			optimizer.step()

		# Update training stats
		train_running_loss += train_loss.item() * train_inputs.size(0)
		train_running_mse += train_mse.item()
		print(f"Training Step: {idx} of {num_train_steps}", end='\r')


	# Store training stats
	train_loss_stat = train_running_loss / len(train_loader.dataset)
	train_mse_stat = train_running_mse / len(train_loader.dataset)

	
	# CUDA cleanup
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


	# VALIDATION PHASE --------
	model.train(mode=False)

	# Init accumulators
	valid_running_loss = 0.0
	valid_running_mse = 0

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

		# Update validation stats
		valid_running_loss += valid_loss.item() * valid_inputs.size(0)
		valid_running_mse += valid_mse.item()


	# Store validation stats
	valid_loss_stat = valid_running_loss / len(valid_loader.dataset)
	valid_mse_stat = valid_running_mse / len(valid_loader.dataset)

	# CUDA cleanup
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		
	# compute metrics.
	print(f'Epoch [{epoch + 1}/{num_epochs}]')
	print(f'[Training]. Loss: {train_loss_stat}, MSE: {train_mse_stat}.')
	print(f'[Validation]. Loss: {valid_loss_stat}, MSE: {valid_mse_stat}.')
	
	# save current model --> replaced at each epoch.
	print("saving model state ...")
	torch.save(
		model.state_dict(), 
		epoch_model_path.format(epoch=epoch, loss=valid_loss_stat, mse=valid_mse_stat)
	)
    
