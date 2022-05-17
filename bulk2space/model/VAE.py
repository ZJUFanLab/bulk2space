# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init
from torch.autograd import Variable

from math import exp


class VAE(nn.Module):
    def __init__(self, embedding_size, hidden_size_list: list, mid_hidden):
        super(VAE, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size_list = hidden_size_list
        self.mid_hidden = mid_hidden

        self.enc_feature_size_list = [self.embedding_size] + self.hidden_size_list + [self.mid_hidden * 2]
        self.dec_feature_size_list = [self.embedding_size] + self.hidden_size_list + [self.mid_hidden]

        self.encoder = nn.ModuleList(
            [nn.Linear(self.enc_feature_size_list[i], self.enc_feature_size_list[i + 1]) for i in
             range(len(self.enc_feature_size_list) - 1)])
        self.decoder = nn.ModuleList(
            [nn.Linear(self.dec_feature_size_list[i], self.dec_feature_size_list[i - 1]) for i in
             range(len(self.dec_feature_size_list) - 1, 0, -1)])

    def encode(self, x):
        for i, layer in enumerate(self.encoder):
            x = self.encoder[i](x)
            if i != len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def decode(self, x):
        for i, layer in enumerate(self.decoder):
            x = self.decoder[i](x)
            # if i != len(self.decoder) - 1:  # 考虑去除，防止负数出现
            x = F.relu(x)
        return x

    def forward(self, x, used_device):
        x = x.to(used_device)

        # import pdb
        # pdb.set_trace()

        encoder_output = self.encode(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)  # mu, log_var
        hidden = torch.randn_like(sigma) + mu * torch.exp(sigma) ** 0.5  # var => std
        x_hat = self.decode(hidden)
        kl_div = 0.5 * torch.sum(torch.exp(sigma) + torch.pow(mu, 2) - 1 - sigma) / (x.shape[0] * x.shape[1])
        return x_hat, kl_div

    def get_hidden(self, x):
        encoder_output = self.encode(x)
        mu, sigma = torch.chunk(encoder_output, 2, dim=1)  # mu, log_var
        hidden = torch.randn_like(sigma) + mu * torch.exp(sigma) ** 0.5  # var => std
        return hidden


class BetaVAE_H(VAE):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, embedding_size, hidden_size_list: list, mid_hidden):
        super(BetaVAE_H, self).__init__(embedding_size, hidden_size_list, mid_hidden)
        self.mid_hidden = mid_hidden
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, used_device):
        x = x.to(used_device)
        distributions = self.encode(x)
        mu = distributions[:, :self.mid_hidden]
        logvar = distributions[:, self.mid_hidden:]
        # mu, logvar = torch.chunk(distributions, 2, dim=1) 
        z = reparametrize(mu, logvar)
        x_recon = self.decode(z)
        total_kld = kl_divergence(mu, logvar)
        return x_recon, total_kld


def kl_divergence(mu, logvar):
    # total_kld = 0.5 * torch.sum(torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar) / (x.shape[0] * x.shape[1])

    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
