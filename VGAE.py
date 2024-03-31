import dgl.function as fn
import torch
import torch.nn.functional as F
from torch import nn
from gcn import GCN


class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features):
        features = self.conv(features)
        return features


class Decoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Decoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features):
        features = self.conv(features)
        return features


class VGAE(nn.Module):
    def __init__(self, g, in_feats, n_hidden, z_dim, n_layers, activation, dropout):
        super(VGAE, self).__init__()

        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        self.decoder = Decoder(g, n_hidden, in_feats, n_layers, activation, dropout)

        # VGAE Loss Computation
        self.z_dim = z_dim
        self.linear_rep = nn.Linear(n_hidden, z_dim)  # mu, log_var
        self.linear_rec = nn.Linear(z_dim, n_hidden)


    def reparameterize(self, mu, log_var):
        # A function that samples from the latent distribution using the reparameterization trick
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std    # return z sample

    def forward(self, features):
        embeddings = self.encoder(features)
        # VAE Decode Part: compute the latent distribution parameters and sample from it
        mu, log_var = self.linear_rep(embeddings), self.linear_rep(embeddings)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decoder(self.linear_rec(z))

        reconst_loss = F.mse_loss(x_reconst, features, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        vgae_loss = reconst_loss + kl_div

        return vgae_loss


class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes, dropout=0.2):
        super(Classifier, self).__init__()
        self.mid_feat = None
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, int(n_hidden / 2)),
            nn.ReLU(),
            nn.Linear(int(n_hidden / 2), int(n_hidden / 4)),
            nn.ReLU(),
            nn.Linear(int(n_hidden / 4), n_classes)
        )
        # self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self, nn.Conv2d) or isinstance(self, nn.Linear):
            self.reset_parameters()
        # self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc1(features)
        # self.mid_feat = features
        # features = self.fc2(features)
        return torch.log_softmax(features, dim=-1)
