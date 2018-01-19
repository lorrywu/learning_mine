#coding:utf-8
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import util
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from model import  *
import time
import math

use_cuda = torch.cuda.is_available() # 如果您的计算机支持cuda，则优先在cuda下运行
input_lang, output_lang, pairs = util.prepareData('eng', 'fra', True)
MAX_LENGTH = 30

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(util.EOS_TOKEN)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

teacher_forcing_ratio = 0.5

hidden_size = 256
learning_rate = 0.005
encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               1, dropout_p=0.1)
encoder_optimizer = torch.optim.Adam(encoder1.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(attn_decoder1.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

def train(input_variable, target_variable, encoder, decoder, encoder_opitmizer,
          decoder_optimizer, criterion, max_length = util.MAX_LENGTH):
    #初始化编码器的隐藏层状态
    encoder_hidden = encoder.initHidden()
    decoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #待翻译句子和已翻译句子的长度
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    # 建立一个编码器输出的PyTorch变量，注意命名是-s结尾，表示
    # 该变量保存了Encoder每一次中间状态数据，而不是最后一次中间状态。
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    # 如果使用cuda，则再包装一下
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        #通过解码过程得到编码器每一次的中间状态数据
        encoder_outputs[ei] = encoder_output[0][0]

        #给解码器准备最初的输入,是一个开始占位符
        decoder_input = Variable(torch.LongTensor([[util.SOS_TOKEN]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        #解码器初始的输入就是编码器最后一次中间层状态数据
        decoder_hidden = encoder_hidden

        #该变量表明是否再每一次输出时都用模板正确输出计算损失
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            #条件为真时, 使用正确的输出作为下一刻解码器的输入来循环计算
            for di in range(target_length):
                #decoder解码器具体实施的过程, 确定其输出、隐藏层状态、注意力数据,
                #decoder的forward方法会动态确定decoder_attention数据
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                #更新损失
                loss += criterion(decoder_output, target_variable[di])
                #确定下一时间步的解码器输入
                decoder_input = target_variable[di]

        else:
            #条件不为真时, 使用解码器自身预测的输入作为下一个时刻解码器的输入来询计算
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += criterion(decoder_output, target_variable[di])
                if ni == util.EOS_TOKEN:
                    break

        #反向传递损失
        loss.backward()
        #更新整个网络的参数
        encoder_opitmizer.step()
        decoder_optimizer.step()

        return loss.data[0]/target_length




def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=10, plot_every=100, learning_rate=0.01):
    start = time.time() # 启动计时
    plot_losses = []    # 保存需要绘制的loss
    print_loss_total = 0  # Reset every print_every 设置loss采样频率
    plot_loss_total = 0  # Reset every plot_every
    # 声明两个RNN的优化器
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # 得到训练使用的数据
    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    # 损失计算方法
    criterion = nn.NLLLoss()

    # 循环训练，迭代的次数
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, (iter+0.0) / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)



def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    # 把sentence转化为网络可以接受的输入，同时初始化编码器的隐藏状态
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()
    # 准备编码器输出变量
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # 得到编码的输出
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    # 准备解码器输出的变量
    decoder_input = Variable(torch.LongTensor([[util.SOS_TOKEN]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # 编码器和解码器之间的桥梁:Context
    decoder_hidden = encoder_hidden
    # 准备一个列表来保存网络预测的词语
    decoded_words = []
    # 准备一个变量保存解码过程中产生的注意力数据
    decoder_attentions = torch.zeros(max_length, max_length)

    # 解码过程，有一个最大长度限制
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)

        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == util.EOS_TOKEN:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        # 解码器的输出作为其输入
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    # 返回预测的单词，以及注意力机制（供分析注意力机制）
    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

if __name__ == "__main__":
    # 核心的训练代码仅此一句
    trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
    #evaluate
    evaluateRandomly(encoder1, attn_decoder1)