import torch
import torch.nn.functional as F

from kws.models import CRNNBase


class CRNNStreaming(CRNNBase):
    
    def __init__(self, config):
        super().__init__(config)

    def inference(self, x, hidden=None):
        self.buffer = torch.cat((self.buffer, x), dim=2)[:, :, -self.config.max_window_length:]

        input = self.buffer.unsqueeze(dim=1)
        conv_output = self.conv(input).transpose(-1, -2)
        gru_output, hidden = self.gru(conv_output, hidden)
        contex_vector = self.attention(gru_output)
        logits = self.classifier(contex_vector)

        probs = F.softmax(logits, dim=-1)

        return probs.detach().cpu(), hidden
    
    def set_buffer(self, x):
        self.buffer = torch.zeros(size=(1 if len(x.shape) == 2 else x.shape[0], 
                                        self.config.num_mels, 
                                        self.config.max_window_length), 
                                  device=self.config.device)