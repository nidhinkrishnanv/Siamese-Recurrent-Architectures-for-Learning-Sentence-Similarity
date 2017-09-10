import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class MaLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_out=50):
        self.embed_layer = nn.Embedding(vocab_size, dim)
        self.lstm = [nn.LSTM(dim, lstm_out), nn.LSTM(dim, lstm_out)]

    def forward(self, inputs, seq_sizes, original_order):
        #First Network
        embed = [self.embed_layer(inputs[x]) for x in range(2)]
        packed = [pack_padded_sequence(embed[x]) for x in range(2)]

        lstm_out = []; out = []
        for x in range(2)
            lstm_out[x], _ = self.lstm[x](packed[x], hidden)
            out[x], _ = pad_packed_sequence(lstm_out[x], batch_first=True)