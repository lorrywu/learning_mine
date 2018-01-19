#coding:utf-8

import math
import random
import time

import matplotlib.pyplot as plt
import torch
import  torch.nn as nn
from torch.autograd import Variable

import model
from  com.lorry.nlp.names_classify import data

all_categories = data.all_categories
category_lines = data.category_lines
n_categories = len(all_categories)
n_letters = data.n_letters
all_letters = data.all_letters

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def targetTensor(line):
    letter_indexs = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexs.append(n_letters -1)
    return torch.LongTensor(letter_indexs)

def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = Variable(categoryTensor(category))
    input_line_tensor = Variable(inputTensor(line))
    target_line_tensor = Variable(targetTensor(line))
    return category_tensor, input_line_tensor, target_line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

criterion = nn.NLLLoss()
learning_rate = 0.0005
input_size = n_letters;
hidden_size = 128
output_size = n_letters
rnn = model.RNN(input_size, hidden_size, output_size)

def trainOneBatch(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i]) #sum up the loss
    loss.backward() #一句话计算所有梯度

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data) #根据梯度更新的相关权重参数

    return output, loss.data[0]/input_line_tensor.size()[0]

def train():
    total_loss = 0
    all_losses = []
    plot_every = 500
    print_every = 5000
    n_iters = 100000
    start = time.time()
    for iter in range(1, n_iters + 1):
        category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
        output, loss = trainOneBatch(category_tensor, input_line_tensor, target_line_tensor)
        total_loss += loss
        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / (n_iters + 0.0) * 100, loss))
        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    torch.save(rnn, "generate.ptk")
    plt.figure()
    plt.plot(all_losses)

if __name__ == "__main__":
    train()






