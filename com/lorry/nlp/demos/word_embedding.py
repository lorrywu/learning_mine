import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch import  optim
import math

import torch.nn.functional  as F

def similarity(a, b):
    # commen = 0
    # a_mode = 0;
    # b_mode = 0;
    print a
    print b

    commen = torch.mul(a, b)
    coo = 0.0
    for co in commen.data[0]:
        coo += co
    a_length = 0.0
    b_length = 0.0
    for data in a.data[0]:
        a_length += data*data
    for data in b.data[0]:
        b_length += data*data

    return coo/(math.sqrt(a_length) * math.sqrt(b_length))



        # i = i[1].data[0]
        # j = j[1].data[0]
        # commen += i.data * j
        # a_mode += i * i
        # b_mode += j * j

    # a_mode = math.sqrt(a_mode)
    # b_mode = math.sqrt(b_mode)
    #
    # return commen/(a_mode * b_mode)



word_to_ix = {'hello': 0, 'hi': 1}
embeds = nn.Embedding(2, 10)
#access the embedding
hello_idx = torch.LongTensor([word_to_ix['hello']])
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)

hi_idx = torch.LongTensor([word_to_ix['hi']])
hi_idx = Variable(hi_idx)
hi_embed = embeds(hi_idx)

#print("hello: ", hello_embed)
#print("hi: ", hi_embed)

print ("~~~~~~~~~~~~~~~")
print (similarity(hi_embed, hello_embed))

# for i in enumerate(hello_embed):
#     index, vari = i
#     print (index)
#     print(vari.data)
#     vari = vari.data
#     print ("~~~~~~~~~")
#     for j in enumerate(vari):
#         print(j[1])

