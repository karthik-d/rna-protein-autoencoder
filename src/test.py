import torch

from nn.simple_autoencoder import AutoEncoder
from data.covid_dataset import CovidDataset


print("GPU Access:", torch.cuda.is_available())
model_path = "../data/models_subset/epoch-20_mse-2.6286497116088867_loss-473.0475158691406.pth"

# Model Initialization
device = 'cpu'
# device = 'cuda:0'
model = AutoEncoder().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
 
# Validation using MSE Loss function
mse_function = torch.nn.MSELoss()


def get_data_loader():
	
	dataset = CovidDataset(split='test')
	# wrap dataset into dataloader.
	return torch.utils.data.DataLoader(
		dataset,
		batch_size=32,
		shuffle=False
	)


num_epochs = 50
output =[]
data_loader = get_data_loader()

loss_l = []
mse_l = []
corr_l = []
with torch.no_grad():

	for data in data_loader:
		rna, protein = data

		# to gpu.
		rna = rna.to(device)
		protein_recon = model(rna)

		# compute mse.
		mse = mse_function(protein_recon, protein)
		mse_l.append(mse)

		corr = torch.corrcoef(torch.cat([protein_recon.flatten(), protein.flatten()]))
		# print(type(protein_recon.flatten()))
		corr_l.append(corr)

# compute metrics.
avg_mse = sum(mse_l)/len(mse_l)
avg_corr = sum(corr_l)/len(corr_l)

print(f'Avg. MSE :{avg_mse}')
print(f'Avg. Corr:{avg_corr}')
    
