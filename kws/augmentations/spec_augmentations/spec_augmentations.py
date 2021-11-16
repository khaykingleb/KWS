import torchaudio

import torch
from torch import nn


class LogMelSpec:

    def __init__(self, is_train, config):
        # With augmentations
        if is_train:
            self.melspec = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=config.sample_rate,
                    n_fft=config.n_fft,
                    win_length=config.win_length,
                    hop_length=config.hop_length,
                    n_mels=config.num_mels
                ),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=config.freq_mask),
                torchaudio.transforms.TimeMasking(time_mask_param=config.time_mask),
            ).to(config.device)

        # No augmentations
        else:
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.win_length,
                hop_length=config.hop_length,
                n_mels=config.n_mels
            ).to(config.device)

    def __call__(self, batch):
        return torch.log(self.melspec(batch).clamp_(min=1e-9, max=1e9))
