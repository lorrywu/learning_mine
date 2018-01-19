import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

a = torch.rand(5, 4)
b = a.numpy()
print (a)
print (a.size())
print b

a = np.array([[3, 4], [3, 6]])
b = torch.from_numpy(a)
print a
print b

x = Variable(torch.Tensor([3]), requires_grad = True)
y = Variable(torch.Tensor([5]), requires_grad = True)
z = 2 * x * x + y*y + 4

z.backward()

print (x.grad.data)

print ('dz/dx:{}'.format(x.grad.data))
print ('dz/dx:{}'.format(y.grad.data))


print torch.cuda.is_available()
class net_name(nn.Module):
    def __init__(self):
        super(net_name, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 3)

    def forward(self, x):
        out = self.conv1(x)
        return out
