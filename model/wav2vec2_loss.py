import random

import torch
from torch import nn

from model.config import Config


class Wav2vec2Loss(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.k = config.contrastive_loss_temperature
        self.K = config.num_contrastive_loss_negative_samples
        self.cos = nn.CosineSimilarity(dim=-1)
        self.G = config.num_code_vector_groups
        self.V = config.num_code_vectors_per_group
        self.a = config.loss_alpha

    def forward(self, encoder_out, quantized_features, perplexity, time_mask_indices):
        target_encoder_out = encoder_out[time_mask_indices]
        labels = quantized_features[time_mask_indices]

        # Number of targets per batch
        num_targets_per_batch = [int(time_mask_indices[i].sum()) for i in range(time_mask_indices.size(0))]

        # Make negative samples
        negative_samples = self.negative_sampler(labels, num_targets_per_batch)
        negative_samples = torch.cat([labels.unsqueeze(1), negative_samples], dim=1)

        contrastive_loss = self.contrastive_loss(target_encoder_out, labels, negative_samples)
        diversity_loss = self.diversity_loss(perplexity)

        loss = contrastive_loss + self.a * diversity_loss

        return loss

    def contrastive_loss(
            self,
            targets: torch.Tensor,
            labels: torch.Tensor,
            negative_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            targets (torch.Tensor): with shape `(N, D)`
            labels (torch.Tensor): with shape `(N, D)`
            negative_samples (torch.Tensor): with shape `(N, K, D)`

        Returns:
            torch.Tensor with shape `(1)`
        """

        similarity = torch.exp(self.cos(targets, labels) / self.k)
        negative_similarity = torch.sum(torch.exp((self.cos(targets.unsqueeze(1), negative_samples) / self.k)), dim=1)

        contrastive_loss = -torch.log(similarity / negative_similarity).mean()

        return contrastive_loss

    def diversity_loss(self, perplexity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            perplexity (torch.Tensor): with shape `(G, V)`

        Returns:
            torch.Tensor with shape `(1)`
        """
        log_perplexity = torch.log(perplexity)
        entropy = torch.sum(perplexity*log_perplexity, dim=-1)
        diversity_loss = torch.sum(entropy) / (self.G * self.V)

        return diversity_loss

    def negative_sampler(self, label: torch.Tensor, num_targets_per_batch: list[int]):
        """
        Args:
            label (torch.Tensor): with shape `(N, D)`
            num_targets_per_batch (list[int]): Number of targets per batch.

        Returns:
            torch.Tensor with shape `(N, K, D)'

        """
        negative_samples = []
        start_idx = 0
        for num_targets in num_targets_per_batch:
            negative_sample_candidate_indices = torch.arange(
                num_targets, device=label.device
            ).unsqueeze(0).repeat(num_targets, 1)

            diagonal = torch.eye(num_targets)

            # Pull yourself from the list of candidates. `(N, N)` -> `(N, N-1)`
            negative_sample_candidate_indices = negative_sample_candidate_indices[diagonal == 0].view(num_targets, -1)
            negative_sample_candidate_indices += start_idx

            where_negative_sample = (
                torch.tensor([i for i in range(num_targets) for _ in range(self.K)]),
                torch.tensor(
                    [random.sample(list(range(num_targets - 1)), k=self.K) for _ in range(num_targets)]).flatten()
            )

            # `(K * N)`
            negative_sample_indices = negative_sample_candidate_indices[where_negative_sample]

            negative_samples.append(label[negative_sample_indices])
            start_idx += num_targets

        negative_samples = torch.cat(negative_samples).view(label.size(0), self.K, -1)

        return negative_samples
