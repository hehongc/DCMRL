import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from simpl.nn import TensorBatch


def sample_positive_pairs(n_task, enc_size, buffers, device):

    task_indices = np.random.randint(len(buffers), size=n_task)

    query_enc_batches = []
    key_enc_batches = []

    for task_idx in task_indices:
        query_enc_batch = buffers[task_idx].sample(enc_size).to(device)
        key_enc_batch = buffers[task_idx].sample(enc_size).to(device)

        query_enc_batches.append(query_enc_batch)
        key_enc_batches.append(key_enc_batch)

    query_states = torch.stack([query_enc_batch.states for query_enc_batch in query_enc_batches], dim=0)
    query_actions = torch.stack([query_enc_batch.actions for query_enc_batch in query_enc_batches], dim=0)
    query_rewards = torch.stack([query_enc_batch.rewards for query_enc_batch in query_enc_batches], dim=0)
    query_dones = torch.stack([query_enc_batch.dones for query_enc_batch in query_enc_batches], dim=0)
    query_next_states = torch.stack([query_enc_batch.next_states for query_enc_batch in query_enc_batches], dim=0)

    query_rewards = query_rewards.unsqueeze(-1)
    query_dones = query_dones.unsqueeze(-1)

    key_states = torch.stack([query_enc_batch.states for query_enc_batch in key_enc_batches], dim=0)
    key_actions = torch.stack([query_enc_batch.actions for query_enc_batch in key_enc_batches], dim=0)
    key_rewards = torch.stack([query_enc_batch.rewards for query_enc_batch in key_enc_batches], dim=0)
    key_dones = torch.stack([query_enc_batch.dones for query_enc_batch in key_enc_batches], dim=0)
    key_next_states = torch.stack([query_enc_batch.next_states for query_enc_batch in key_enc_batches], dim=0)

    key_rewards = key_rewards.unsqueeze(-1)
    key_dones = key_dones.unsqueeze(-1)

    # positive_Batch = TensorBatch(
    #     positive_task_indices=task_indices,
    #
    #     query_states=query_states,
    #     query_actions=query_actions,
    #     query_rewards=query_rewards,
    #     query_dones=query_dones,
    #     query_next_states=query_next_states,
    #
    #     key_states=key_states,
    #     key_actions=key_actions,
    #     key_rewards=key_rewards,
    #     key_dones=key_dones,
    #     key_next_states=key_next_states,
    # )

    # return positive_Batch
    return task_indices, \
           query_states, query_actions, query_rewards, query_dones, query_next_states, \
           key_states, key_actions, key_rewards, key_dones, key_next_states


def sample_negative_pairs(positive_task_indices, n_task, enc_size, buffers, device):

    task_indices = range(len(buffers))
    task_indices = set(task_indices)

    positive_task_indices = set(positive_task_indices)

    task_indices = list(task_indices - positive_task_indices)
    task_indices = np.random.choice(task_indices, size=n_task, replace=True)

    negative_batches = []
    for task_idx in task_indices:
        negative_batch = buffers[task_idx].sample(enc_size).to(device)
        negative_batches.append(negative_batch)

    negative_states = torch.stack([negative_enc_batch.states for negative_enc_batch in negative_batches], dim=0)
    negative_actions = torch.stack([negative_enc_batch.actions for negative_enc_batch in negative_batches], dim=0)
    negative_rewards = torch.stack([negative_enc_batch.rewards for negative_enc_batch in negative_batches], dim=0)
    negative_dones = torch.stack([negative_enc_batch.dones for negative_enc_batch in negative_batches], dim=0)
    negative_next_states = torch.stack([negative_enc_batch.next_states for negative_enc_batch in negative_batches],
                                       dim=0)

    negative_dones = negative_dones.unsqueeze(-1)
    negative_rewards = negative_rewards.unsqueeze(-1)

    negative_states = negative_states.reshape(n_task, enc_size, -1)
    negative_actions = negative_actions.reshape(n_task, enc_size, -1)
    negative_rewards = negative_rewards.reshape(n_task, enc_size, -1)
    negative_dones = negative_dones.reshape(n_task, enc_size, -1)
    negative_next_states = negative_next_states.reshape(n_task, enc_size, -1)

    # negative_Batch = TensorBatch(
    #     negative_states=negative_states,
    #     negative_actions=negative_actions,
    #     negative_rewards=negative_rewards,
    #     negative_dones=negative_dones,
    #     negative_next_states=negative_next_states,
    # )

    # return negative_Batch
    return negative_states, negative_actions, negative_rewards, negative_dones, negative_next_states


def sample_positive_transitions(n_task, enc_size, buffers, device):

    task_indices = np.random.randint(len(buffers), size=n_task)

    query_enc_transitions = []
    key_enc_transitions = []

    for task_idx in task_indices:
        query_enc_batch = buffers[task_idx].sample(enc_size).to(device)
        query_enc_transition = query_enc_batch.as_transitions()
        query_enc_transitions.append(query_enc_transition)

        key_enc_batch = buffers[task_idx].sample(enc_size).to(device)
        key_enc_transition = key_enc_batch.as_transitions()
        key_enc_transitions.append(key_enc_transition)

    query_enc_transitions = torch.stack(query_enc_transitions, dim=0)
    key_enc_transitions = torch.stack(key_enc_transitions, dim=0)

    # return positive_Batch
    return task_indices, query_enc_transitions, key_enc_transitions


def sample_positive_transitions_new(n_task, enc_size, buffers, device):

    task_indices = np.random.randint(len(buffers), size=n_task)

    query_enc_transitions = []
    key_enc_transitions = []

    for task_idx in task_indices:
        buffer_size = buffers[task_idx].size
        indices = torch.randint(buffer_size, size=[enc_size * 2])

        query_indices = torch.randint(enc_size*2, size=[enc_size])
        query_indices = torch.sort(query_indices)[0]
        key_indices = torch.randint(enc_size* 2, size=[enc_size])
        key_indices = torch.sort(key_indices)[0]

        query_enc_batch_indices = indices[query_indices]
        query_enc_batch = buffers[task_idx].sample_with_indices(query_enc_batch_indices).to(device)

        key_enc_batch_indices = indices[key_indices]
        key_enc_batch = buffers[task_idx].sample_with_indices(key_enc_batch_indices).to(device)

        query_enc_transition = query_enc_batch.as_transitions().to(device)
        query_enc_transitions.append(query_enc_transition)

        key_enc_transition = key_enc_batch.as_transitions().to(device)
        key_enc_transitions.append(key_enc_transition)

    query_enc_transitions = torch.stack(query_enc_transitions, dim=0).to(device)
    key_enc_transitions = torch.stack(key_enc_transitions, dim=0).to(device)

    # return positive_Batch
    return task_indices, query_enc_transitions, key_enc_transitions


def sample_negative_transitions(positive_task_indices, n_task, enc_size, buffers, device):

    task_indices = range(len(buffers))
    task_indices = set(task_indices)

    positive_task_indices = set(positive_task_indices)

    task_indices = list(task_indices - positive_task_indices)
    task_indices = np.random.choice(task_indices, size=n_task, replace=True)

    negative_enc_transitions = []
    for task_idx in task_indices:
        negative_batch = buffers[task_idx].sample(enc_size).to(device)
        negative_transition = negative_batch.as_transitions()
        negative_enc_transitions.append(negative_transition)

    negative_enc_transitions = torch.stack(negative_enc_transitions, dim=0)

    # return negative_Batch
    return negative_enc_transitions


# InfoNCE
# q, k (b, dim); neg (b, N, dim)
def contrastive_loss(q, k, neg, n_task, batch_size, n_negative_per_positive, device, infonce_temp):
    N = n_negative_per_positive
    b = n_task * batch_size

    l_pos = torch.bmm(q.view(b, 1, -1), k.view(b, -1, 1))  # (b,1,1)
    l_neg = torch.bmm(q.view(b, 1, -1), neg.transpose(1, 2))  # (b,1,N)
    logits = torch.cat([l_pos.view(b, 1), l_neg.view(b, N)], dim=1)  # (b, N+1)

    labels = torch.zeros(b, dtype=torch.long)
    labels = labels.to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()
    loss = cross_entropy_loss(logits / infonce_temp, labels)

    return loss


def TripletLoss(q, k, neg, device):
    margin = 1.0
    task_num = q.shape[0]
    neg_num = neg.shape[0]

    # (task_num)
    pos_cos_sim = F.cosine_similarity(q, k, dim=-1).to(device)

    q_expand = q.unsqueeze(0).expand(neg_num, task_num, q.shape[-1])
    # (neg_num, task_num) -> (task_num)
    neg_cos_sim = F.cosine_similarity(q_expand, neg, dim=-1).mean(dim=0).to(device)

    loss = torch.mean(torch.clamp(neg_cos_sim - pos_cos_sim + margin, min=0.0))

    return loss


