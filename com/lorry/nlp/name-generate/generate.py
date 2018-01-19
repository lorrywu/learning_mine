#coding:utf-8
import torch
from torch.autograd import Variable

import train

max_length = 20 #设定一个生成人名的最大长度
rnn = torch.load("generate.ptk")
def sample(category, start_letter = 'W'):
    category_tensor = Variable(train.categoryTensor(category))
    input = Variable(train.inputTensor(start_letter))
    hidden = rnn.initHidden()
    output_name = start_letter

    for i in range(max_length):
        output, hidden = rnn(category_tensor, input[0], hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == train.n_letters - 1:
            break
        else:
            letter = train.all_letters[topi]
            output_name += letter
        input = Variable(train.inputTensor(letter))

    return output_name

def samples(category, start_letters = "ABC"):
    for start_letter in start_letters:
        print sample(category, start_letter)

if __name__ == "__main__":
    name = samples("Chinese")
    print name
