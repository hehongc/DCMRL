import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F

from simpl.math import inverse_softplus, inverse_sigmoid
from simpl.nn import MLP, SetTransformer
from simpl.alg.spirl.VQ_embedding import VQEmbedding



class StochasticEncoder(nn.Module):
    def __init__(self, z_dim, prior_scale):
        super().__init__()
        self.register_buffer('prior_loc', torch.zeros(z_dim))
        self.register_buffer('prior_scale', prior_scale*torch.ones(z_dim))

        self.device = None

    def to(self, device):
        self.device = device
        return super().to(device)
     
    @property
    def prior_dist(self):
        return torch_dist.Independent(torch_dist.Normal(self.prior_loc, self.prior_scale), 1)
    
    def dist(self, batch_transitions):
        raise NotImplementedError

    def encode(self, list_batch, sample):
        if len(list_batch) == 0:
            dist = self.prior_dist

            if sample is True:
                z = dist.sample()
            else:
                z = dist.mean
        else:
            transitions = torch.cat([batch.as_transitions() for batch in list_batch], dim=0)
            with torch.no_grad():
                dist = self.dist(transitions.to(self.device).unsqueeze(0))
        
            if sample is True:
                z = dist.sample().squeeze(0)
            else:
                z = dist.mean.squeeze(0)
        return z.cpu().numpy()


def product_of_gaussians(mus, sigma_squares):
    sigma_squares = sigma_squares.clamp(min=1e-7)

    sigma_square = 1 / (1 / sigma_squares).sum(-2)
    mu = sigma_square * (mus / sigma_squares).sum(-2)

    return mu, sigma_square


class GaussianProductEncoder(StochasticEncoder):
    def __init__(self, state_dim, action_dim, z_dim, hidden_dim, n_hidden, min_scale=0.001, init_scale=0.1):
        super().__init__(z_dim, init_scale)
        self.min_scale = min_scale
        self.rff = MLP([2*state_dim + action_dim + 2] + [hidden_dim]*n_hidden + [2*z_dim])
        self.pre_init_scale = float(np.log(np.exp(init_scale) - 1))  # inverse softplus

    def dist(self, batch_transitions):
        batch_mus, batch_pre_sigma_squares = self.rff(batch_transitions).chunk(2, dim=-1)
        batch_sigma_squares = self.min_scale + F.softplus(self.pre_init_scale + batch_pre_sigma_squares)
        batch_mu, batch_sigma_square = product_of_gaussians(batch_mus, batch_sigma_squares)
        return torch_dist.Independent(torch_dist.Normal(batch_mu, batch_sigma_square.sqrt()), 1)


class DeepSetEncoder(StochasticEncoder):
    def __init__(self, state_dim, action_dim, z_dim, hidden_dim, n_hidden, min_scale=0.001, init_scale=1, prior_scale=1):
        super().__init__(z_dim, prior_scale)
        self.min_scale = min_scale
        self.pre_net = MLP([2*state_dim + action_dim + 2] + [hidden_dim]*n_hidden)
        self.post_net = MLP([hidden_dim]*n_hidden + [2*z_dim])
        self.pre_init_scale = float(np.log(np.exp(init_scale) - 1))  # inverse softplus

    def dist(self, batch_transitions):
        batch_hs = self.pre_net(batch_transitions)
        batch_h = batch_hs.mean(1)
        batch_locs, batch_pre_scales = self.post_net(batch_h).chunk(2, dim=-1)
        batch_scales = self.min_scale + F.softplus(self.pre_init_scale + batch_pre_scales)
        return torch_dist.Independent(torch_dist.Normal(batch_locs, batch_scales), 1)


class SetTransformerEncoder(StochasticEncoder):
    def __init__(self, state_dim, action_dim, z_dim,
                 hidden_dim, n_hidden, activation='relu', codebook_size=10, beta=0.25,
                 min_scale=0.001, max_scale=None, init_scale=1, prior_scale=1):
        super().__init__(z_dim, prior_scale)

        self.net = SetTransformer(
            2*state_dim + action_dim + 2, 2*z_dim,
            hidden_dim, n_hidden, n_hidden, activation=activation
        )
        self.min_scale = min_scale
        self.max_scale = max_scale

        if max_scale is None:
            self.pre_init_scale = inverse_softplus(init_scale)
        else:
            self.pre_init_scale = inverse_sigmoid(max_scale/init_scale - 1)

        # VQ-VAE part
        self.codebook_size = int(codebook_size)
        self.code_size = 2 * z_dim
        self.beta = beta
        self.vq = VQEmbedding(self.codebook_size, self.code_size, self.beta)

        self.decoder = MLP([self.code_size] + [hidden_dim] * n_hidden + [hidden_dim], activation)

        # mse loss
        self.mse_loss = nn.MSELoss(reduction='none')

    def dist(self, batch_transitions):
        loc, pre_scale = self.net(batch_transitions).chunk(2, dim=-1)
        if self.max_scale is None:
            scale = self.min_scale + F.softplus(self.pre_init_scale + pre_scale)
        else:
            scale = self.min_scale + self.max_scale*torch.sigmoid(self.pre_init_scale + pre_scale)
        return torch_dist.Independent(torch_dist.Normal(loc, scale), 1)


    def get_value(self, batch_transitions):
        loc, pre_scale = self.net(batch_transitions).chunk(2, dim=-1)
        if self.max_scale is None:
            scale = self.min_scale + F.softplus(self.pre_init_scale + pre_scale)
        else:
            scale = self.min_scale + self.max_scale * torch.sigmoid(self.pre_init_scale + pre_scale)

        output = torch.cat([loc, scale], dim=-1)

        return output


    def dist_with_value(self, batch_transitions):
        output, mlp_input = self.net.forward_with_mlp_input(batch_transitions)
        loc, pre_scale = output.chunk(2, dim=-1)

        if self.max_scale is None:
            scale = self.min_scale + F.softplus(self.pre_init_scale + pre_scale)
        else:
            scale = self.min_scale + self.max_scale * torch.sigmoid(self.pre_init_scale + pre_scale)

        dist_value = torch.cat([loc, scale], dim=-1)
        return torch_dist.Independent(torch_dist.Normal(loc, scale), 1), dist_value, mlp_input


    def reconsitution(self, batch_action):
        reconsitution_state = self.decoder(batch_action)
        return reconsitution_state

    def code_match(self, batch_action):
        codes_no_grad, select_codes = self.vq.straight_through(batch_action)
        return codes_no_grad, select_codes

    def compute_logprob(self, batch_state, reconsitution_state):
        logprob = -1. * self.mse_loss(batch_state, reconsitution_state)
        return logprob.mean(dim=-1)

    def compute_loss(self, batch_action, select_codes, logprob):
        loss = self.vq(batch_action, select_codes).mean(dim=-1) - logprob
        return loss