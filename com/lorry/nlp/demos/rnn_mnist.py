from torchvision import transforms
from torchvision import datasets
from torch.utils.data import  DataLoader
import torch
from torch import nn, optim
import torch.nn.functional  as F
from torch.autograd import Variable
import time

batch_size = 32
learning_rate = 0.001
num_epoches = 1000

train_dataset = datasets.MNIST(root='/Users/lorry/Documents/python/pytorch_learn_be/my_data/MNIST', train=True,
                               transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(root='/Users/lorry/Documents/python/pytorch_learn_be/my_data/MNIST',
                              train=False, transform=transforms.ToTensor())


train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

class RNNNetword(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(RNNNetword, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classfier = nn.Linear(hidden_dim, n_class)


    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classfier(out)
        return out

model = RNNNetword(28, 128, 2, 10)


#if has gpu
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

#training
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch))
    print ('*'*10)
    running_loss = 0.0
    running_acc = 0.0
    since = time.time()
    for i, data in enumerate(train_loader, 1):
        img, label = data
        b, c, h, w = img.size()
        assert  c == 1
        img = img.squeeze(1)
        #no need to expand
        #img = img.view(img.size(0), -1) # expand image to 28 * 28, batch_size * 748

        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        #forwoard
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]

        #back forward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
                train_dataset))))

model.eval()
eval_loss = 0.
eval_acc = 0.
for data in test_dataset:
    img, label = data
    b, c, h, w = img.size()
    assert c == 1, 'channel must be 1'
    img = img.squeeze(1)

    if use_gpu:
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
    else:
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.data[0]
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
print('Time:{:.1f} s'.format(time.time() - since))
print()

torch.save(model.state_dict(), './neural_mnist.pth')
