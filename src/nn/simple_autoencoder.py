import torch

class AutoEncoder(torch.nn.Module):
    
	def __init__(self, n_latent_space, n_input_size=2000, output_activation='linear'):
		super().__init__()
			
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# n_input_size ==> n_latent_space
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(n_input_size, 1024),
			torch.nn.BatchNorm1d(1024),
			torch.nn.ReLU(),
			torch.nn.Linear(1024, 256),
			torch.nn.BatchNorm1d(256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 128),
			torch.nn.BatchNorm1d(128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, n_latent_space)
		)
			
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# n_latent_space ==> 138
		decoder_modules = [
			torch.nn.Linear(n_latent_space, 32),
			torch.nn.BatchNorm1d(32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 64),
			torch.nn.BatchNorm1d(64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 138)
		]

		# decide activation function.
		if output_activation == 'linear':
			pass 
		elif output_activation == 'sigmoid':
			decoder_modules.append(torch.nn.Sigmoid())

		# make decoder.
		self.decoder = torch.nn.Sequential(*decoder_modules)
		

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded, encoded
