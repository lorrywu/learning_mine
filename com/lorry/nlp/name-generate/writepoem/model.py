#coding:utf-8
"""
write poem
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layer = 1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layer = n_layer
        self.encoder = nn.Embedding(input_size, hidden_size) # 1 * 1 * hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layer)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layer, 1, self.hidden_size))


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = nn.Embedding(input_size, hidden_size)  # 1 * 1 * hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def forward(self, input, h):
        input = self.encoder(input.view(1, -1))
        output, h = self.lstm(input, h)
        output = output.squeeze(0)
        output = self.decoder(output)
        return output, h

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_size)),
                Variable(torch.zeros(1, 1, self.hidden_size)))

class LSTM_1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def forward(self, input, h):
        input = input.view(1, 1, -1)
        output, h = self.lstm(input, h)
        output = output.squeeze(0)
        output = self.decoder(output)
        return output, h

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_size)),
                Variable(torch.zeros(1, 1, self.hidden_size)))