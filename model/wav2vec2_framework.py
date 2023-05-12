import random

import torch
from torch import nn

from model import Config
from model.gumbel_vector_quantizer import GumbelVectorQuantizer


class Wav2Vec2Framework(nn.Module):
    def __init__(self, config: Config, feature_extractor: nn.Module, encoder: nn.Module):
        super().__init__()
        self.mask_time_prob = config.mask_time_prob
        self.num_mask_time_steps = config.num_mask_time_steps

        self.feature_extractor: nn.Module = feature_extractor
        self.encoder: nn.Module = encoder

        self.quantizer = GumbelVectorQuantizer(config)
        self.out_linear = nn.Linear(
            config.encoder_hidden_size, config.code_vector_size
        )

    def forward(self, input_values: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            input_values (torch.Tensor): with shape `(B, T, D1)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            hidden_states (torch.Tensor): with shape `(B, L, D2)`
            quantized_features (torch.Tensor): with shape `(B, L, D2)`
            perplexity (torch.Tensor): with shape `(G, V)`
            time_mask (torch.BoolTensor): with shape `(B, L)`
            )
        """

        hidden_states, lengths = self.feature_extractor(input_values, lengths)
        masked_hidden_states, time_mask_indices = self.time_masking(hidden_states, lengths)

        quantized_features, perplexity = self.quantizer(hidden_states, lengths)

        encoder_out, _ = self.encoder(masked_hidden_states, lengths)

        encoder_out = self.out_linear(encoder_out)

        return encoder_out, quantized_features, perplexity, time_mask_indices

    def time_masking(self, hidden_states: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.BoolTensor]:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            lengths (torch.Tensor): with shape `(B)`

        Returns:
            tuple(
            Masked hidden states (torch.Tensor with shape `(B, L, D)`),
            Time mask (torch.BoolTensor with `(B, L)`)
            )
        """

        batch_size, num_steps, hidden_size = hidden_states.size()

        # non mask: 0, mask: 1
        time_mask_indices = torch.zeros(
            batch_size, num_steps + self.num_mask_time_steps,
            device=hidden_states.device, dtype=torch.bool
        )

        for batch in range(batch_size):
            time_mask_idx_candidates = list(range(int(lengths[batch])))
            k = int(self.mask_time_prob * lengths[batch])
            start_time_idx_array = torch.tensor(
                random.sample(time_mask_idx_candidates, k=k), device=hidden_states.device
            )

            for i in range(self.num_mask_time_steps):
                time_mask_indices[batch, start_time_idx_array+i] = 1

        time_mask_indices = time_mask_indices[:, :-self.num_mask_time_steps]
        num_masks = sum(time_mask_indices.flatten())

        # Maks hidden states
        mask_values = torch.zeros(num_masks, hidden_size, device=hidden_states.device)
        hidden_states[time_mask_indices] = mask_values

        return hidden_states, time_mask_indices
