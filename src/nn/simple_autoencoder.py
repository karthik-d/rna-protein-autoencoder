import torch

class AutoEncoder(torch.nn.Module):
    
	def __init__(self, n_latent_space):
		super().__init__()
			
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 3000 ==> n_latent_space
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(2000, 1024),
			torch.nn.ReLU(),
			torch.nn.Linear(1024, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, n_latent_space)
		)
			
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# n_latent_space ==> 138
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(n_latent_space, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 138) #,
			# torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
