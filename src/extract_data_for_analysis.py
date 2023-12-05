import pandas as pd
import numpy as np
import torch
import pathlib
import os

from nn.simple_autoencoder import AutoEncoder
from data.covid_dataset import CovidDataset
from utils.metrics import compute_colwise_correlations, compute_colwise_spearman_correlations
from utils.plots import save_line_plots


# SET WEIGHT FILE ------------------
WEIGHTFILE_PATHS = [
	"../data/models/run_run_lr-1.00e-01_dr-1.00e-02_ls-8_inp-norm/epoch-1_corr-nan_loss-0.061.pth"
]
NORMALIZATION_METHOD = 'minmax'
OUTPUT_ACTIVATION = 'sigmoid'
# ----------------------------------


# init GPU.
gpu_is_available = torch.cuda.is_available()
device = torch.device("cuda:0" if gpu_is_available else "cpu")
print("GPU Access:", gpu_is_available)
print("DEVICE:", device)
if gpu_is_available:
	print("DEVICE Name:", torch.cuda.get_device_name(0))


# configure paths.
output_root = "../data/output"
output_dir_recipe = os.path.join(output_root, "{run_name}")

pathlib.Path(output_root).mkdir(
	exist_ok = True,
	parents = True
)


# get data loaders.
def get_data_loaders(batch_size, input_type, normalization_method, verbose=True):

	test_dataset = CovidDataset(version='two', split='test', input_type=input_type, normalization_method=normalization_method)
	valid_dataset = CovidDataset(version='two', split='valid', input_type=input_type, normalization_method=normalization_method)
	# set verbosity
	test_dataset.set_verbose_render(state=verbose)
	valid_dataset.set_verbose_render(state=verbose)
	# wrap dataset into dataloader.
	return torch.utils.data.DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=True,
		collate_fn=test_dataset.collation_register_metadata
	), torch.utils.data.DataLoader(
		valid_dataset,
		batch_size=batch_size,
		shuffle=False,
		collate_fn=valid_dataset.collation_register_metadata
	)


def extraction_job(
	n_latent_space,
	run_combination_str,
	output_activation,
	test_loader,
	valid_loader,
	weight_path
):

	# lazy creation; create output saving path.
	pathlib.Path(output_dir_recipe.format(run_name=run_combination_str)).mkdir(
		parents = True,
		exist_ok = True
	)
	
	model = AutoEncoder(
		n_latent_space=n_latent_space,
		output_activation=output_activation
	).to(device)

	
	# load model; set grad to false.
	torch.load(weight_path)
	model.train(mode=False)

	
	# init validation accumulators.
	valid_labels_batchwise = []
	valid_predictions_batchwise = []
	valid_latentspace_batchwise = []

	# feed forward over all the validation data.
	for k in valid_loader:
		print(k)
		break
	for idx, (valid_inputs, valid_labels) in enumerate(valid_loader):
		valid_inputs = valid_inputs.to(device=device)
		valid_labels = valid_labels.to(device=device)
		
		# Feed-Forward ONLY!
		with torch.set_grad_enabled(mode=False):
			valid_outputs, valid_latent_repr = model(valid_inputs)

		valid_labels_batchwise.append(pd.DataFrame(valid_labels, columns=valid_y_cols, index=valid_rows))
		valid_predictions_batchwise.append(pd.DataFrame(valid_outputs, columns=valid_y_cols, index=valid_rows))
		valid_latentspace_batchwise.append(pd.DataFrame(valid_latent_repr, index=valid_rows))

	# CUDA cleanup.
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	# save outputs.
	print("saving validation extracts ...")
	pd.concatenate(valid_labels_batchwise).to_csv(os.path.join(
		output_dir_recipe.format(run_name=run_combination_str),
		"validation-labels.tsv"
	), sep='\t')
	pd.concatenate(valid_predictions_batchwise).to_csv(os.path.join(
		output_dir_recipe.format(run_name=run_combination_str),
		"validation-predictions.tsv"
	), sep='\t')
	pd.concatenate(valid_latentspace_batchwise).to_csv(os.path.join(
		output_dir_recipe.format(run_name=run_combination_str),
		"validation-latentspace.tsv"
	), sep='\t')


	# init test accumulators.
	test_labels_batchwise = []
	test_predictions_batchwise = []
	test_latentspace_batchwise = []

	# feed forward over all the test data.
	for idx, (test_inputs, test_labels), (test_x_cols, test_y_cols), test_rows in enumerate(test_loader):
		test_inputs = test_inputs.to(device=device)
		test_labels = test_labels.to(device=device)
		
		# Feed-Forward ONLY!
		with torch.set_grad_enabled(mode=False):
			test_outputs, test_latent_repr = model(test_inputs)

		test_labels_batchwise.append(pd.DataFrame(test_labels, columns=test_y_cols, index=test_rows))
		test_predictions_batchwise.append(pd.DataFrame(test_outputs, columns=test_y_cols, index=test_rows))
		test_latentspace_batchwise.append(pd.DataFrame(test_latent_repr, index=test_rows))

	# CUDA cleanup.
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	# save outputs.
	print("saving test extracts ...")
	pd.concatenate(test_labels_batchwise).to_csv(os.path.join(
		output_dir_recipe.format(run_name=run_combination_str),
		"test-labels.tsv"
	), sep='\t')
	pd.concatenate(test_predictions_batchwise).to_csv(os.path.join(
		output_dir_recipe.format(run_name=run_combination_str),
		"test-predictions.tsv"
	), sep='\t')
	pd.concatenate(test_latentspace_batchwise).to_csv(os.path.join(
		output_dir_recipe.format(run_name=run_combination_str),
		"test-latentspace.tsv"
	), sep='\t')


for weight_path in WEIGHTFILE_PATHS:

	# load weight parts.
	epoch_str = os.path.basename(weight_path.split('_')[0])
	param_str = pathlib.Path(os.path.dirname(weight_path)).parts[-1].lstrip("RUNrun_")

	# extract params from string.
	param_parts = param_str.split('_')
	inp_type = param_parts[-1].split('-')[1]
	n_latent_space = int(param_parts[-2].split('-')[1])

	# get loaders.
	# NOTE: set batch_size > 1; not cofigured for singular inputs.
	test_loader, valid_loader = get_data_loaders(
		batch_size = 1,
		input_type = inp_type,
		normalization_method = NORMALIZATION_METHOD
	)

	# run job.
	extraction_job(
		weight_path = weight_path,
		run_combination_str = f"{param_str}_{epoch_str}",
		valid_loader = valid_loader,
		test_loader = test_loader,
		output_activation = OUTPUT_ACTIVATION,
		n_latent_space = n_latent_space
	)