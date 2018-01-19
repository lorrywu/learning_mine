#coding:utf-8

"""
用bp做mnist识别
"""
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import  DataLoader
from com.lorry.dl.fc import *
from datetime import datetime

batch_size = 1

train_dataset = datasets.MNIST(root='/Users/lorry/Documents/python/learn_dl_zero/data/mnist', train=True,
                               transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(root='/Users/lorry/Documents/python/learn_dl_zero/data/mnist',
                              train=False, transform=transforms.ToTensor())


train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

def show(sample):
    str = ''
    for i in range(28):
        for j in range(28):
            if sample[i*28+j] != 0:
                str += '*'
            else:
                str += ' '
        str += '\n'
    print (str)


def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)

    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)


def now():
    return datetime.now().strftime('%c')


def train_and_evaluate():
    last_error_ratio = 0.0001
    epoch = 0

    train_data_set = []
    train_labels = []
    test_data_set =[]
    test_labels = []
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1) # expand image to 28 * 28, batch_size * 748

        train_data_set.append(np.transpose(img.numpy()))
        train_labels.append(np.transpose(label.numpy()))

    for i, data in enumerate(test_dataset, 1):
        img, label = data
        img = img.view(img.size(0), -1)  # expand image to 28 * 28, batch_size * 748
        test_data_set.append(np.transpose(img.numpy()))
        test_labels.append(np.transpose(label.numpy()))


    network = Network([784, 100, 10])
    while (epoch < 100):
        epoch += 1
        network.train(train_labels, train_data_set, 0.01, 1)
        print ('%s epoch %d finished, loss %f' % (now(), epoch,
            network.loss(train_labels[-1], network.predict(train_data_set[-1]))))
        if epoch % 2 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print ('%s after epoch %d, error ratio is %f' % (now(), epoch, error_ratio))
            # if error_ratio > last_error_ratio:
            #     break
            # else:
            #     last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()