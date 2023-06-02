
import torch
import torch.nn as nn

class baseline(nn.Module):
    def __init__(self, num_channels = 52, num_classes = 22):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_channels = self.num_channels, out_channels = 64, kernel_size = 3, stride = 2), 
            nn.LeakyReLU(), 
            nn.BatchNorm1d(num_features = 64, eps = 1e-5, momentum = 0.1), 
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2), 
            nn.LeakyReLU(), 
            nn.BatchNorm1d(num_features = 128, eps = 1e-5, momentum = 0.1), 
        )

        self.encoder3 = nn.Sequential(
            nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1), 
            nn.LeakyReLU(), 
            nn.BatchNorm1d(num_features = 256, eps = 1e-5, momentum = 0.1), 
        )

        self.dense = nn.Sequential(
            nn.Flatten(), 
            # nn.LazyLinear(out_features = self.num_classes), 
            nn.Linear(in_features = 5632, out_features = self.num_classes), 
            nn.Softmax(dim = 1)
        )
    
    def forward(self, X):
        latent = self.encoder1(X)
        latent = self.encoder2(latent)
        latent = self.encoder3(latent)

        y_hat = self.dense(latent)
        return y_hat
        