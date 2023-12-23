
### import packages
import torch
from nn.simple_autoencoder import AutoEncoder
from data.covid_dataset import CovidDataset


### Model Initialization

# init saving path
epoch_model_path = "../data/models/epoch-{epoch}_mse-{mse}_loss-{loss}.pth"

# init GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU Access:", torch.cuda.is_available())
print("DEVICE:", device)
if torch.cuda.is_available():
  print("DEVICE Name:", torch.cuda.get_device_name(0))

model = AutoEncoder().to(device)
 
# define loss function 
loss_function = torch.nn.CrossEntropyLoss()
mse_function = torch.nn.MSELoss()
 
# define optimizer
optimizer = torch.optim.Adam(
	model.parameters(),
    lr = 1e-3,
    weight_decay = 1e-3
)

# load data tool
def get_data_loader():
	
	dataset = CovidDataset(split='train')
	# wrap dataset into dataloader.
	return torch.utils.data.DataLoader(
		dataset,
		batch_size=32,
		shuffle=True
	)


### train model 

num_epochs = 50
output =[]
data_loader = get_data_loader()
for epoch in range(num_epochs):
	print(f"{epoch+1} of {num_epochs} ...")

	loss_l = []
	mse_l = []
	for data in data_loader:
		rna, protein = data

		# to gpu.
		rna = rna.to(device) 
		protein = protein.to(device)
		protein_recon = model(rna)

		# compute loss.                            
		loss = loss_function(protein_recon, protein)
		loss_l.append(loss)

		# compute mse.
		mse = mse_function(protein_recon, protein)
		mse_l.append(mse)
		
		# backpropagate. 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
	
	# compute metrics.
	avg_mse = sum(mse_l)/len(mse_l)
	avg_loss = sum(loss_l)/len(loss_l)
	print(f'Epoch [{epoch + 1}/{num_epochs}], loss:{avg_loss}, MSE:{avg_mse}')
	output.append((epoch, sum(loss_l)/len(loss_l), sum(mse_l)/len(mse_l)))
	
	# save current model --> replaced at each epoch.
	print("saving model state ...")
	torch.save(model.state_dict(), epoch_model_path.format(epoch=epoch, loss=avg_loss, mse=avg_mse))
    
