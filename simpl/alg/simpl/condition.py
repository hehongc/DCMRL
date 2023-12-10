import copy

import torch
import torch.nn as nn

from simpl.rl import StochasticNNPolicy


class ConditionedPolicy(StochasticNNPolicy):
    def __init__(self, policy, z):
        super().__init__()
        self.policy = copy.deepcopy(policy)
        self.z = z

    def dist(self, batch_state):
        batch_z = torch.tensor(self.z, device=self.policy.device)[None, :].expand(len(batch_state), -1)
        batch_state_z = torch.cat([batch_state, batch_z], dim=-1)
        return self.policy.dist(batch_state_z)

    def dist_with_value(self, batch_state):
        batch_z = torch.tensor(self.z, device=self.policy.device)[None, :].expand(len(batch_state), -1)
        batch_state_z = torch.cat([batch_state, batch_z], dim=-1)
        return self.policy.dist_with_value(batch_state_z)

    def code_match(self, batch_action):
        return self.policy.code_match(batch_action)

    def reconsitution(self, batch_action):
        return self.policy.reconsitution(batch_action)

    def compute_logprob(self, batch_state, reconsitution_state):
        return self.policy.compute_logprob(batch_state, reconsitution_state)

    def compute_loss(self, batch_action, select_codes, logprob):
        return self.policy.compute_loss(batch_action, select_codes, logprob)


class ConditionedQF(nn.Module):
    def __init__(self, qf, z):
        super().__init__()
        self.qf = copy.deepcopy(qf)
        self.z = z

    def forward(self, batch_state, batch_action):
        batch_z = torch.tensor(self.z, device=self.qf.device)[None, :].expand(len(batch_state), -1)
        batch_state_z = torch.cat([batch_state, batch_z], dim=-1)
        return self.qf(batch_state_z, batch_action)

