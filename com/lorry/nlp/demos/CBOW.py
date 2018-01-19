import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch import  optim

import torch.nn.functional  as F

#N-gram , N = 3
CONTEXT_SIZE = 4
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty\\'s field,
Thy youth\\'s proud livery so gazed on now,
Will be a totter\\'d weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv\\'d thy beauty\\'s use,
If thou couldst answer \\'This fair child of mine
Shall sum my count, and make my old excuse,\\'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel\\'st it cold.""".split()


trigram = [((test_sentence[i], test_sentence[i+1], test_sentence[i+3], test_sentence[i+4]), test_sentence[i+2]) for i in range(len(test_sentence)-4)]
print trigram

vocb = set(test_sentence)
print vocb
word_to_idx = {word: i for i, word in enumerate(vocb)}
print word_to_idx
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
print idx_to_word

class NgramModel(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(CONTEXT_SIZE * n_dim, 128)
        #self.linear2 = nn.Linear(128, self.n_word)
        self.linear2 = nn.Linear(128, n_dim)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        #log_pro = F.log_softmax(out)
        return out

#def model, criterion and optimizer
ngram_model = NgramModel(len(word_to_idx), CONTEXT_SIZE, 100)
criterion = nn.NLLLoss()
optimizer = optim.SGD(ngram_model.parameters(), lr= 0.0001)

epoch_num = 200

for epoch in range(epoch_num):
    print "epoch : ", epoch
    running_loss = 0
    #for each trigram
    for data in trigram:
        word, label = data
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        #forward
        out = ngram_model(word)
        loss = criterion(out, label)
        running_loss = loss.data[0]

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print "loss is " , running_loss/len(word_to_idx)

word, label = trigram[0]
print word, label
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
print "embedding: ", word
out = ngram_model(word)
_, predict_label = torch.max(out, 1)
#print predict_label.data, "-", predict_label.data[0], "-", predict_label.data[0][0]
predict_word = idx_to_word[predict_label.data[0][0]]
print label, predict_word

