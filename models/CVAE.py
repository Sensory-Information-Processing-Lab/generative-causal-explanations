"""
    Simple convolutional VAE, suitable for MNIST experiments
"""

import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, z_dim, c_dim, x_dim,
                 filt_per_layer=64): # x_dim : total number of pixels
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(int(c_dim), filt_per_layer, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(filt_per_layer, filt_per_layer, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(filt_per_layer, filt_per_layer, 4, stride=1, padding=0),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(int(filt_per_layer*x_dim/16), z_dim)
        self.fc_logvar = nn.Linear(int(filt_per_layer*x_dim/16), z_dim)
    
    def encode(self, x):
        print("CVAE input shape: ", x.shape)
        z = self.model(x)
        print("Z shape: ", z.shape)
        z = z.view(z.shape[0], -1)
        return self.fc_mu(z), self.fc_logvar(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self, z_dim, c_dim, x_dim, # x_dim : total number of pixels
                 filt_per_layer=64):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim, int(filt_per_layer*self.x_dim/16)),
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(filt_per_layer, filt_per_layer, 4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(filt_per_layer, filt_per_layer, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filt_per_layer, int(c_dim), 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.shape[0]
        t = self.fc(z).view(batch_size, -1,
                            int(np.sqrt(self.x_dim)/4),
                            int(np.sqrt(self.x_dim)/4))
        x = self.model(t)
        return x