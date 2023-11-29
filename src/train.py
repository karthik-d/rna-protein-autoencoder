import torch

from nn.simple_autoencoder import AutoEncoder


# Model Initialization
device = 'cuda:0'
model = AutoEncoder().to(device)
 
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)


num_epochs = 100
output =[]
for epoch in range(num_epochs):
    for data in loader:
        img, _ = data
        img = img.reshape(-1,64*64)
        img    = img.to(device)
        recon = model(img)                             
        loss   = loss_function(recon, img.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss. Item(): .4f}')
    output.append((epoch,img,recon))