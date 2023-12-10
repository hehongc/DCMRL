import torch
import torch.nn as nn
from .functions import vector_quantization, vector_quantization_st


class VQEmbedding(nn.Module):
    def __init__(self, codebook_size, code_size, beta):

        super().__init__()

        self.codebook_size = codebook_size
        self.code_size = code_size
        self.beta = beta

        self.embedding = nn.Embedding(self.codebook_size, self.code_size)
        self.embedding.weight.data.uniform_(-1./self.codebook_size, 1./self.codebook_size)

        self.mse_loss = nn.MSELoss(reduction='none')


    def quantize(self, skill):
        return vector_quantization(skill, self.embedding.weight)

    def straight_through(self, skill):
        codes_no_grad, indices = vector_quantization_st(skill, self.embedding.weight.detach())
        select_codes = torch.index_select(self.embedding.weight, dim=0, index=indices)

        return codes_no_grad, select_codes

    def forward(self, skill, select_codes=None):
        if select_codes is None:
            _, select_codes = self.straight_through(skill)

        vq_loss = self.mse_loss(select_codes, skill.detach())
        commitment_loss = self.mse_loss(skill, select_codes.detach())

        loss = vq_loss + self.beta * commitment_loss

        return loss

    def compute_distances(self, inputs):
        with ttorch.no_grad():
            embedding_size = self.embedding.weight.size(1)
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(self.embedding.weight ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, self.embedding.weight.t(),
                                    alpha=-2.0, beta=1.0)

            return distances
