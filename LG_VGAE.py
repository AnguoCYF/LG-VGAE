import dgl.function as fn
import scipy
import sympy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import math

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)

        self.lin = lin
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d + 1):
            inv_coeff.append(float(coeff[d - i]))
        thetas.append(inv_coeff)
    return thetas


class Encoder(nn.Module):
    def __init__(self, g, in_feats, h_feats, d):
        super(Encoder, self).__init__()
        self.g = g
        self.in_feats = in_feats
        self.h_feats = h_feats
        # self.z_dim = int(h_feats/2)
        self.thetas = calculate_theta2(d=d)
        self.act = nn.ReLU()
        self.conv = []
        self.linear1 = nn.Linear(in_feats, h_feats)
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(in_feats, h_feats, self.thetas[i], lin=False))
        self.linear2 = nn.Linear(h_feats * len(self.conv), h_feats)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.linear1(features)
        features = self.act(features)
        features_final = torch.zeros([len(features), 0]).to(features.device)
        for conv in self.conv:
            h0 = conv(self.g, features)
            features_final = torch.cat([features_final, h0], -1)

        features = self.linear2(features_final)

        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class Decoder(nn.Module):
    def __init__(self, g, in_feats, h_feats, d):
        super(Decoder, self).__init__()
        self.g = g
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.thetas = calculate_theta2(d=d)
        self.act = nn.ReLU()
        self.conv = []
        self.linear1 = nn.Linear(in_feats, h_feats)
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(in_feats, h_feats, self.thetas[i], lin=False))
        self.linear2 = nn.Linear(h_feats * len(self.conv), h_feats)

    def forward(self, features):
        features = self.linear1(features)
        features = self.act(features)
        features_final = torch.zeros([len(features), 0]).to(features.device)
        for conv in self.conv:
            h0 = conv(self.g, features)
            features_final = torch.cat([features_final, h0], -1)
        features = self.linear2(features_final)

        return features


class LG_VGAE(nn.Module):
    def __init__(self, g, in_feats, n_hidden, z_dim, d, b=0.2):
        super(LG_VGAE, self).__init__()

        self.encoder = Encoder(g, in_feats, n_hidden, d)
        self.decoder = Decoder(g, n_hidden, in_feats, d)

        # VGAE Loss Computation
        self.z_dim = z_dim
        self.linear_rep = nn.Linear(n_hidden, z_dim)  # mu, log_var
        self.linear_rec = nn.Linear(z_dim, n_hidden)

        # DGI Loss Computation
        self.discriminator = Discriminator(n_hidden)
        self.lossDGI = nn.BCEWithLogitsLoss()

        # Joint loss balance constant
        self.b = b
        # self.dropout = nn.Dropout(p=dropout)

    def reparameterize(self, mu, log_var):
        # A function that samples from the latent distribution using the reparameterization trick
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std    # return z sample

    def forward(self, features):
        b = self.b
        # features = self.dropout(features)
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)

        # Compute DGI loss using binary cross entropy
        summary = torch.sigmoid(positive.mean(dim=0))
        positive_dis = self.discriminator(positive, summary)
        negative_dis = self.discriminator(negative, summary)
        loss_dgi_1 = self.lossDGI(positive_dis, torch.ones_like(positive_dis))
        loss_dgi_2 = self.lossDGI(negative_dis, torch.zeros_like(negative_dis))
        dgi_loss = loss_dgi_1 + loss_dgi_2

        # VAE Decode Part: compute the latent distribution parameters and sample from it
        mu, log_var = self.linear_rep(positive), self.linear_rep(positive)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decoder(self.linear_rec(z))

        reconst_loss = F.mse_loss(x_reconst, features, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        vgae_loss = reconst_loss + kl_div

        # Joint loss computation: use balance constant b to weigh VGAE and DGI losses
        joint_loss = b * dgi_loss / (dgi_loss/vgae_loss).detach() + (1 - b) * vgae_loss

        return joint_loss


class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes, dropout=0.3):
        super(Classifier, self).__init__()
        self.mid_feat = None
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, int(n_hidden / 2)),
            nn.ReLU(),
            nn.Linear(int(n_hidden / 2), int(n_hidden / 4))
        )
        self.fc2 = nn.Sequential(
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
        self.mid_feat = features
        features = self.fc2(features)
        return torch.log_softmax(features, dim=-1)
