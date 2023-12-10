from simpl.rl import ContextPolicyMixin
from simpl.alg.spirl import PriorResidualNormalMLPPolicy

from simpl.nn import MLP
from simpl.alg.spirl.VQ_embedding import VQEmbedding

import torch
import torch.distributions as torch_dist
import torch.nn.functional as F
import torch.nn as nn



class ContextPriorResidualNormalMLPPolicy(ContextPolicyMixin, PriorResidualNormalMLPPolicy):
    def __init__(self, prior_policy, state_dim, action_dim, z_dim, hidden_dim, n_hidden,
                 codebook_size=10, beta=0.25, prior_state_dim=None, policy_exclude_dim=None, activation='relu'):
        if prior_state_dim is None:
            prior_state_dim = state_dim
        super().__init__(
            prior_policy, state_dim + z_dim, action_dim, hidden_dim, n_hidden, 
            prior_state_dim, policy_exclude_dim, activation
        )
        self.z_dim = z_dim

        self.codebook_size = int(codebook_size)
        self.code_size = action_dim * 2
        self.beta = beta
        self.vq = VQEmbedding(self.codebook_size, self.code_size, self.beta)

        self.decoder = MLP([self.code_size] + [hidden_dim]*n_hidden + [state_dim + z_dim], activation)

        self.mse_loss = nn.MSELoss(reduction='none')



    def reconsitution(self, batch_action):
        self.prior_policy.eval()
        reconsitution_state = self.decoder(batch_action)
        return reconsitution_state

    def code_match(self, batch_action):
        self.prior_policy.eval()
        codes_no_grad, select_codes = self.vq.straight_through(batch_action)
        return codes_no_grad, select_codes

    def compute_logprob(self, batch_state, reconsitution_state):
        logprob = -1. * self.mse_loss(batch_state, reconsitution_state)
        return logprob.mean(dim=1)

    def compute_loss(self, batch_action, select_codes, logprob):
        loss = self.vq(batch_action, select_codes).mean(dim=1) - logprob
        return loss






