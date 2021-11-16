import torchaudio
import torch
from torch import Tensor

from configs.config import Config


class WaveAugs:

    def __init__(self, config: Config):
        self.background_noises = [
            'KWS/data/speech_commands/_background_noise_/white_noise.wav',
            'KWS/data/speech_commands/_background_noise_/dude_miaowing.wav',
            'KWS/data/speech_commands/_background_noise_/doing_the_dishes.wav',
            'KWS/data/speech_commands/_background_noise_/exercise_bike.wav',
            'KWS/data/speech_commands/_background_noise_/pink_noise.wav',
            'KWS/data/speech_commands/_background_noise_/running_tap.wav'
        ]

        self.config = config

        self.noises = [torchaudio.load(p)[0].squeeze() for p in self.background_noises]

    def add_rand_noise(self, audio):
        # Randomly choose noise
        noise_num = torch.randint(low=0, high=len(self.background_noises), size=(1,)).item()
        noise = self.noises[noise_num]

        noise_level = torch.Tensor([1])  # [0, 40]

        noise_energy = torch.norm(noise)
        audio_energy = torch.norm(audio)
        alpha = (audio_energy / noise_energy) * torch.pow(10, -noise_level / 20)

        start = torch.randint(low=0, high=max(int(noise.size(0) - audio.size(0) - 1), 1),
                              size=(1,)).item()

        noise_sample = noise[start: start + audio.size(0)]

        audio_new = audio + alpha * noise_sample
        audio_new.clamp_(-1, 1)

        return audio_new

    def __call__(self, wav):
        # Choose one random augmentation
        aug_num = torch.randint(low=0, high=4, size=(1,)).item()

        augs = [
            lambda x: x,
            lambda x: (x + torch.distributions.Normal(0, self.config.noise_std).sample(x.size())).clamp_(-1, 1),
            lambda x: torchaudio.transforms.Vol(self.config.gain_vol, gain_type="amplitude")(x),
            lambda x: self.add_rand_noise(x)
        ]

        return augs[aug_num](wav)
