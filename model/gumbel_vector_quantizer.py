import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from model.wav2vec2_config import Wav2Vec2Config


class GumbelVectorQuantizer(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.num_groups = config.num_code_vector_groups
        self.num_vectors = config.num_code_vectors_per_group

        self.linear = nn.Linear(
            config.extracted_feature_size,
            self.num_groups * self.num_vectors
        )
        self.code_book = nn.Parameter(
            torch.FloatTensor(1, self.num_groups, self.num_vectors, config.code_vector_size // self.num_groups)
        )

        self.temperature = config.gumbel_init_temperature

    def _compute_perplexity(self, probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs (torch.Tensor): with shape `(B, L, G, V)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            torch.Tensor with shape `(G, V)`
        """
        where_calculate_probs = torch.arange(probs.size(1), device=probs.device).unsqueeze(0) < lengths.unsqueeze(-1)
        probs = probs[where_calculate_probs == 1]

        num_values = probs.size(0)
        perplexity = probs.sum(0) / num_values

        return perplexity

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D1)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            code_vectors (torch.Tensor): with shape `(B, L, D2)`
            perplexity (torch.Tensor): with shape `(G, V)`
            )
        """

        batch_size, length, _ = hidden_states.shape

        hidden_states = self.linear(hidden_states)
        # `(B, L, G * V)` -> `(B * L * G, V)`
        hidden_states = hidden_states.view(batch_size * length * self.num_groups, -1)

        code_vector_probs = nn.functional.gumbel_softmax(
            hidden_states.float(), tau=self.temperature, hard=True
        ).type_as(hidden_states)
        code_vector_soft_dist = torch.softmax(
            hidden_states.view(batch_size, length, self.num_groups, -1).float(), dim=-1
        )
        perplexity = self._compute_perplexity(code_vector_soft_dist, lengths)

        code_vector_probs = code_vector_probs.view(batch_size * length, self.num_groups, -1).unsqueeze(-1)

        code_vectors = code_vector_probs * self.code_book
        # `(B * L, G, V, D)` -> `(B, L, G * D)`
        code_vectors = code_vectors.sum(-2).view(batch_size, length, -1)

        return code_vectors, perplexity
