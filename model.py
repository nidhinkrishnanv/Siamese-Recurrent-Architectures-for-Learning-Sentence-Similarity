import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class MaLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=10):
        super(MaLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden = []
        # self.lstm_l = {0:nn.LSTM(embed_dim, self.hidden_dim, batch_first=True), 1:nn.LSTM(embed_dim, self.hidden_dim, batch_first=True)}
        self.lstm_l = [None]*2
        # for i in range(2):
        #     self.lstm_l[i] = nn.LSTM(embed_dim, self.hidden_dim, batch_first=True)

        self.lstm1 = nn.LSTM(embed_dim, self.hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(embed_dim, self.hidden_dim, batch_first=True)


    def forward(self, sentences, sent_lengths, orders):
        #First Network
        embeds = [self.embeddings(sentences[i]) for i in range(2)]
        packed = [pack_padded_sequence(embeds[i], sent_lengths[i], batch_first=True) for i in range(2)]
        lstm_out = [None]*2; out = [None]*2
        
        lstm_out[0], _ = self.lstm1(packed[0], self.hidden[0])
        lstm_out[1], _ = self.lstm2(packed[1], self.hidden[1])
        h_out = [None]*2
        for i in range(2):
            out[i], _ = pad_packed_sequence(lstm_out[i], batch_first=True)
            h_out[i] = torch.gather(out[i], 1, Variable(torch.LongTensor(sent_lengths[i]-1).cuda().view(-1, 1, 1)).expand(out[i].size()[0], 1, self.hidden_dim)).squeeze(1)
            # print(h_out[i])

            # reorder
            # print(orders[i])
            out[i] = torch.gather(h_out[i], 0, orders[i].view(-1,1).expand(h_out[i].size()[0], self.hidden_dim))
            # print(out[i])

        # find the l1 pariwise distance
        l1_norm = F.pairwise_distance(h_out[0], h_out[1], p=1)

        out = torch.exp(-l1_norm)
        return out

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()),
            Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda()))
