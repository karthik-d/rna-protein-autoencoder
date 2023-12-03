import torch

class AutoEncoder(torch.nn.Module):
    
	def __init__(self):
		super().__init__()
			
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 3000 ==> 16
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(2000, 1024),
			torch.nn.ReLU(),
			torch.nn.Linear(1024, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 16)
		)
			
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# 16 ==> 138
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(16, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 138),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
