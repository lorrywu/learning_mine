import torch
import  torch.nn as nn
from torch.autograd import Variable
from com.lorry.nlp.names_classify import data

n_categories = len(data.all_categories)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()


    def forward(self, category, input, hidden):
        input_combine = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combine)
        output = self.i2o(input_combine)
        output_combine = torch.cat((hidden, output), 1)
        output = self.o2o(output_combine)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

