import torch
import numpy as np

import torch.nn as nn
import vectorization as vec
import os

# Define the Variational Autoencoder model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            # nn.Sigmoid(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.Sigmoid(),
            nn.Linear(100, latent_dim * 2), # Two times latent_dim for mean and variance
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 300),
            nn.ReLU(),
            nn.Linear(300, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim),
            nn.Tanh()
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        z_mean_log_var = self.encoder(x)
        z_mean = z_mean_log_var[:, :latent_dim]
        z_log_var = z_mean_log_var[:, latent_dim:]
        
        z = self.reparameterize(z_mean, z_log_var)
        x_hat = self.decoder(z)
        return x_hat, z_mean, z_log_var

# Set the dimensions
n = 17
input_dim = int(0.5*(n**2)*(n**2-1))
latent_dim = 8

# Create the VAE model
model = VAE(input_dim, latent_dim)


# Find all files that start with "225" and end with ".fold"

file_list = []
directory = 'trainingData'
for filename in os.listdir(directory):
    if filename.startswith('225') and filename.endswith('.fold'):
        file_list.append(os.path.join(directory, filename))

# Vectorize and concatenate the data from each file
data_list = []
for file in file_list:
    data = vec.vectorize(file)
    data_list.append(data)

train_data = torch.from_numpy(np.concatenate(data_list, axis=1).T).float()

# # Load the training data
# bird_base = vec.vectorize('trainingData/225bird_base.fold')
# bird_frog = vec.vectorize('trainingData/225bird_frog_base.fold')
# blintzed_bird = vec.vectorize('trainingData/225blintzed_bird_base.fold')
# deerf = vec.vectorize('trainingData/225deerf.fold')
# deerm = vec.vectorize('trainingData/225deerm.fold')
# dragon = vec.vectorize('trainingData/225dragon.fold')
# # empty = vec.vectorize('trainingData/empty.fold')

# train_data = torch.from_numpy(np.concatenate((bird_base,bird_frog,blintzed_bird,deerf,deerm,dragon),axis=1).T).float()
print(train_data.shape)


# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Set the number of epochs
num_epochs = 70
# Set the batch size
batch_size = 128
# Create a data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        x = batch
        x_hat, z_mean, z_log_var = model(x)
        # Compute the reconstruction loss
        reconstruction_loss = nn.MSELoss()(x_hat, x)
        # Compute the negative log likelihood loss
        nll_loss = torch.mean(0.5 * torch.sum((x - x_hat) ** 2, dim=1))
        # Compute the KL divergence loss
        kl_divergence_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        # Compute the total loss
        total_loss = nll_loss + reconstruction_loss + kl_divergence_loss
        # Backward pass
        total_loss.backward()
        # Update the parameters
        optimizer.step()
    # Print the loss for each epoch
    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {round(total_loss.item(),8)}")
        with torch.no_grad():
            x = train_data[0]  # Choose an input from the training data
            latent_values = model.encoder(x.unsqueeze(0))
            print(f"Latent Layer Values: {latent_values[:, :latent_dim].squeeze().tolist()}")


def generate_output(model, latent_dim,scale):
    # Generate random latent vector
    z = torch.rand(1, latent_dim) * scale
    # Decode the latent vector
    output = model.decoder(z)
    return output

output00 = generate_output(model, latent_dim,1).detach().numpy().T
vec.fold2readable(vec.vector2fold(output00/max(output00),0.05),"test00.png")

output0 = generate_output(model, latent_dim,1).detach().numpy().T
vec.fold2readable(vec.vector2fold(output0/max(output0),0.1),"test0.png")

output1 = generate_output(model, latent_dim,1).detach().numpy().T
vec.fold2readable(vec.vector2fold(output1/max(output1),0.2),"test1.png")

output2 = generate_output(model, latent_dim,1).detach().numpy().T
vec.fold2readable(vec.vector2fold(output2/max(output2),0.3),"test2.png")

output3 = generate_output(model, latent_dim,1).detach().numpy().T
vec.fold2readable(vec.vector2fold(output3/max(output3),0.4),"test3.png")

output4 = generate_output(model, latent_dim,1).detach().numpy().T
vec.fold2readable(vec.vector2fold(output4/max(output4),0.5),"test4.png")

output5 = generate_output(model, latent_dim,1).detach().numpy().T
vec.fold2readable(vec.vector2fold(output4/max(output5),0.6),"test5.png")

print("=======finished=======")

#TODO: figure out how to set up NLL loss
