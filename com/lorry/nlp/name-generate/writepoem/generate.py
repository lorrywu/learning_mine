#coding:utf-8
import torch

from util import *
#from model import *
from com.lorry.nlp.utils import get_wordembed

word_embed_dict = get_wordembed.get_word_emb()
def generate(lang, decoder, prime_str= 'A', predict_len = 100, tempertature = 0.8):

    hidden = decoder.init_hidden()

    prime_input = char_tensor(lang, prime_str)
    predicted = prime_str

    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)

    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        output_dist = output.data.view(-1).div(tempertature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        predicted_char = lang.index2char[top_i]
        predicted += predicted_char
        inp= char_tensor(lang, predicted_char)

    return predicted

def generate_embed(lang, decoder, prime_str= 'A', predict_len = 100, tempertature = 0.8):

    hidden = decoder.init_hidden()

    prime_input = Variable(torch.FloatTensor(word_embed_dict[prime_str])).view(1, -1)
    predicted = prime_str

    for p in range(len(prime_str) - 1):
        prime_tensor = Variable(torch.FloatTensor(word_embed_dict[prime_input[p]])).view(1, -1)
        _, hidden = decoder(prime_tensor, hidden)

    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        output_dist = output.data.view(-1).div(tempertature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        predicted_char = lang.index2char[top_i]
        predicted += predicted_char
        if predicted_char in word_embed_dict.keys():
            inp= Variable(torch.FloatTensor(word_embed_dict[predicted_char])).view(1, -1)

    return predicted

if __name__ == '__main__':
    filename = '../data/tangshi.txt'
    model_name = '../data/tangshi.lstm_h50w'
    lang = Lang(filename)
    decoder = torch.load(model_name)
    #start_char = lang.index2char[random.randint(0, len(lang.index2char) - 1)]
    #print start_char
    start_char = u"床疑举低"
    words = ""
    for char in start_char:
        words = generate(lang, decoder, char, predict_len=4)
        print words




