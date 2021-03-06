from typing import Tuple
import dataclasses

import torch


@dataclasses.dataclass
class Config:
    model_name: str = "streaming_crnn"
    seed: int = 100
    verbose: bool = True

    num_workers: int = 2
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_to_data: str = "KWS/data/speech_commands"
    path_to_save: str = "KWS/saved/"
    
    # Data processing: General
    keyword: str = "sheila"
    num_classes: int = 2
    train_ratio: float = 0.8

    # Data processing: Melspectrogram
    sample_rate: int = 16000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    num_mels: int = 40

    # Data processing: Wave augmentations
    noise_std: float = 0.01
    gain_vol: float = 0.25

    # Data processing: Spectrogram augmentations
    freq_mask: int = 15
    time_mask: int = 35

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5

    # Training
    batch_size: int = 128
    num_epochs: int = 10

    # Model
    cnn_out_channels: int = 8
    kernel_size: Tuple[int, int] = (5, 20)
    stride: Tuple[int, int] = (2, 8)
    hidden_size: int = 64
    gru_num_layers: int = 2
    bidirectional: bool = False

    # Streaming
    max_window_length: int = 41
    streaming_step_size: int = 1


@dataclasses.dataclass
class SmallConfig:
    model_name: str = "distilled_crnn"

    num_mels: int = 40
    num_classes: int = 2

    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_to_load: str = "KWS/saved/streaming_crnn_best.pth"

    # Student model
    cnn_out_channels: int = 5
    kernel_size: Tuple[int, int] = (5, 20)
    stride: Tuple[int, int] = (2, 8)
    hidden_size: int = 20
    gru_num_layers: int = 1
    bidirectional: bool = False
    max_window_length: int = 41
    streaming_step_size: int = 1

    # Distillation
    use_distillation: bool = True
    temperature: float = 20.0
    alpha: float = 0.5
