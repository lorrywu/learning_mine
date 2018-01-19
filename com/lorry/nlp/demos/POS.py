import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

training_data = [("The dog ate the apple".split(),
                  ["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(), ["NN", "V", "DET",
                                                       "NN"])]

word_to_idx = {}
tag_to_idx = {}
idx_to_word = {}
idx_to_tag = {}
for context, tag in training_data:
    for word in context:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
            idx_to_word[len(word_to_idx) - 1] = word
    for label in tag:
        if label not in tag_to_idx:
            tag_to_idx[label] = len(tag_to_idx)
            idx_to_tag[len(tag_to_idx) - 1] = label

alphabet = 'abcdefghijklmnopqrstuvwxyz'
character_to_idx = {}
for i in range(len(alphabet)):
    character_to_idx[alphabet[i]] = i

#char lstm
class CharLSTM(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        self.char_embedding = nn.Embedding(n_char, char_dim)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, batch_first=True)

    def forward(self, x):
        #after embeding x size is: 1 *  n_char * char_dim
        x = self.char_embedding(x)

        # h size is 2* n_char * char_hidden
        output, h = self.char_lstm(x)
        return h[1]


class LSTMTagger(nn.Module):
    def __init__(self, n_word, n_char, char_dim, n_dim, char_hidden, n_hidden,
                 n_tag):
        super(LSTMTagger, self).__init__()
        self.word_embedding = nn.Embedding(n_word, n_dim)
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
        self.lstm = nn.LSTM(n_dim + char_hidden, n_hidden, batch_first=True)
        self.linear1 = nn.Linear(n_hidden, n_tag)

    def forward(self, x, word):
        char = torch.FloatTensor()
        for each in word:
            char_list = []
            for letter in each:
                char_list.append(character_to_idx[letter.lower()])

            #print ("before char_list: " , char_list) char_list_size is n_char
            char_list = torch.LongTensor(char_list)
            print ("after char_list: ", char_list)

            #char_list size is 1 *  n_letter
            char_list = char_list.unsqueeze(0)
            print ("unsqueeze(0): ", char_list)

            if torch.cuda.is_available():
                tempchar = self.char_lstm(Variable(char_list).cuda())
            else:
                # tempchar is 1 * 1 * char_hidden
                tempchar = self.char_lstm(Variable(char_list))


            #tempchar after squeeze size is 1 * char_hidden
            tempchar = tempchar.squeeze(0)


            #char size is 1 * char_hidden, char final size is n * char_hidden
            char = torch.cat((char, tempchar.cpu().data), 0)

        if torch.cuda.is_available():
            char = char.cuda()
        char = Variable(char) #n_word * char_hidden
        x = self.word_embedding(x) # n_word * n_dim

        # before  size is : n_word * (char_hidden + n_dim)
        x = torch.cat((x, char), 1)
        x = x.unsqueeze(0) # x size is 1 * 5 * 150, 1 * n_word * (char_hidden + n_dim)

        # x size is
        x, h = self.lstm(x) # x size is 1 * 5 * 128, 1 * n_word * n_hidden
        x = x.squeeze(0) # x size is : 5 * 128, n_word * n_hidden
        x = self.linear1(x) #5 * 5, n_word * n_tag
        y = F.log_softmax(x) # 5 * 3, n_word * n_tag, prop
        return y


model = LSTMTagger(
    len(word_to_idx), len(character_to_idx), 10, 100, 50, 128, len(tag_to_idx))
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)


def make_sequence(x, dic):
    idx = [dic[i] for i in x]
    idx = Variable(torch.LongTensor(idx))
    return idx


for epoch in range(300):
    print('*' * 10)
    print('epoch {}'.format(epoch + 1))
    running_loss = 0
    for data in training_data:
        word, tag = data
        word_list = make_sequence(word, word_to_idx)
        tag = make_sequence(tag, tag_to_idx)
        if torch.cuda.is_available():
            word_list = word_list.cuda()
            tag = tag.cuda()
        # forward
        out = model(word_list, word)
        loss = criterion(out, tag)
        running_loss += loss.data[0]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Loss: {}'.format(running_loss / len(data)))
print()

input = make_sequence("Everybody ate the apple".split(), word_to_idx)
if torch.cuda.is_available():
    input = input.cuda()

words = "Everybody ate the apple".split()
out = model(input, words)

print(out)

_, predict_label = torch.max(out, 1)
print predict_label
#predict_label = predict_label.squeeze(1)

print (idx_to_tag)
print words
index = 0
for x in  words:
    print (x, " ", idx_to_tag[predict_label.data[index][0]])
    index = index + 1

