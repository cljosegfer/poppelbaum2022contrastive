
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class encoder(nn.Module):
    def __init__(self, num_channels = 52):
        super().__init__()
        self.num_channels = num_channels

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
        )
    
    def forward(self, X):
        latent = self.encoder1(X)
        latent = self.encoder2(latent)
        latent = self.encoder3(latent)
        
        flatten = self.dense(latent)

        return flatten

class linear(nn.Module):
    def __init__(self, num_classes = 22):
        super().__init__()
        self.num_classes = num_classes

        self.dense = nn.Sequential(
            nn.Linear(in_features = 5632, out_features = self.num_classes), 
            nn.Softmax(dim = 1)
        )

    def forward(self, X):
        y_hat = self.dense(X)
        return y_hat

def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)

class ContrastiveLoss(nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(self, temperature=0.1):
       super().__init__()
       self.temperature = temperature

   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
    #    return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
       return representations @ representations.transpose(-2, -1)

   def forward(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       batch_size = proj_1.shape[0]
       z_i = F.normalize(proj_1, p=2, dim=1)
       z_j = F.normalize(proj_2, p=2, dim=1)

       similarity_matrix = self.calc_similarity_batch(z_i, z_j)

       sim_ij = torch.diag(similarity_matrix, batch_size)
       sim_ji = torch.diag(similarity_matrix, -batch_size)

       positives = torch.cat([sim_ij, sim_ji], dim=0)

       nominator = torch.exp(positives / self.temperature)
    #    print(batch_size)
       mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
    #    denominator = device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)
       denominator = mask.cuda() * torch.exp(similarity_matrix / self.temperature)

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * batch_size)
       return loss
