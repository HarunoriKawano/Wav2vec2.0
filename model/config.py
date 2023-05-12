from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    code_vector_size: int  # Dimension of quantized code vector (Default: 768)
    num_code_vector_groups: int  # Number of code vector divisions (Default: 2)
    num_code_vectors_per_group: int  # Number of code vectors (Default: 320)
    mask_time_prob: float  # Probability of time masking (Default: 0.065)
    num_mask_time_steps: int  # Number of sequences per code vector (Default: 10)
    extracted_feature_size: int  # Output dimension of feature extractor
    encoder_hidden_size: int  # Output dimension of encoder
    gumbel_init_temperature: int  # Initialized value of gumbel temperature (Default: 2)
    contrastive_loss_temperature: float  # Temperature in contrastive loss (Default: 0.1)
    num_contrastive_loss_negative_samples: int  # Number of negative samples in contrastive loss (Default: 100)
    loss_alpha: float  # loss = contrastive_loss + loss_alpha * diversity_loss (Default: 0.1)
