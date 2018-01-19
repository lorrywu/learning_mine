import torch
each = [1, 2, 3]
dict = {1: 1, 2:2, 3:3}
word_list = []
for letter in each:
    word_list.append(dict[letter])

word_list = torch.FloatTensor(word_list)
print word_list
word_list = word_list.squeeze(0)
print word_list
