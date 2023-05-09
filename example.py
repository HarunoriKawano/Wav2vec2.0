import json

import torch
from torch import nn

from model import Wav2Vec2Framework, Wav2vec2Loss, Wav2Vec2Config


class ExampleFeatureExtractor(nn.Module):
    def __init__(self, extracted_feature_size):
        super().__init__()
        self.linear = nn.Linear(80, extracted_feature_size)

    def forward(self, inputs, lengths):
        hidden_states = self.linear(inputs)

        return hidden_states, lengths


class ExampleEncoder(nn.Module):
    def __init__(self, extracted_feature_size, encoder_hidden_size):
        super().__init__()
        self.linear = nn.Linear(extracted_feature_size, encoder_hidden_size)

    def forward(self, hidden_states, lengths):
        hidden_states = self.linear(hidden_states)

        return hidden_states, lengths


if __name__ == '__main__':
    extracted_feature_size = 562
    encoder_hidden_size = 1024

    feature_extractor = ExampleFeatureExtractor(extracted_feature_size)
    encoder = ExampleEncoder(extracted_feature_size, encoder_hidden_size)

    # `(batch size, time steps, feature size)`
    inputs = torch.randn(4, 1000, 80)
    # `(batch size)` Number of available time steps per batch
    input_lengths = torch.tensor([1000, 871, 389, 487])

    with open("config.json", "r", encoding="utf-8") as f:
        config = Wav2Vec2Config(**json.load(f))

    model = Wav2Vec2Framework(config, feature_extractor, encoder)
    compute_loss = Wav2vec2Loss(config)

    # tuple(Encoder hidden states with mask, Quantized features, Code book perplexity, Time mask indices)
    model_out = model(inputs, input_lengths)
    loss = compute_loss(*model_out)

    print(loss)
