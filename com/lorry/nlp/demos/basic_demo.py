import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch import  optim

import torch.nn.functional  as F

v_data = [1, 2, 3]
v = torch.IntTensor(v_data)

p_data = [44, 22, 33]
p = torch.IntTensor(p_data)

m_data = [[1, 2, 3],[4, 5, 6]]
m = torch.IntTensor(m_data)


n_data = [[1, 2, 3],[4, 5, 6]]
n = torch.IntTensor(m_data)

t_data = [[[1, 2],[3, 4]], [[5, 6],[7, 8]]]
t = torch.IntTensor(t_data)

v = Variable(v)
m = Variable(m)
n = Variable(n)
p = Variable(p)
print v, p


print ("~~~~~~~~~~~~~~~~~~~~~~~")
#enumerate
for value in enumerate(v):
   print value


print ("~~~~~~~~~~~~~~~~~~~~~~~")
#enumerate
for value in enumerate(p):
   print value

print ("~~~~~~~~~~~~~~~~~~~~~~~")


linear = nn.Linear(2, 4)
input = Variable(torch.ones(3, 2))
print input
weight = linear.weight
bias = linear.bias
print "weight", weight
print "bias", bias
output = linear(input)

print "output", output.data


