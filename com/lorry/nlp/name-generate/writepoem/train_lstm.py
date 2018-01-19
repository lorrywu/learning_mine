#coding:utf-8

from model import *
from generate import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from util import *

filename = '../data/tangshi.txt'
hidden_size = 128
n_layer = 1
learning_rate  = 0.005
#chunk_len = 7
epochs = 300000
print_every = 500

lang = read_file(filename)
file = lang.text
print file[0:100]
file_len = lang.text_length
n_characters = len(lang.char2index)
start = time.time()

def random_training_set(chunk_len):
    start_index = random.randint(0, file_len - chunk_len - 1)
    end_index = start_index + chunk_len + 1
    chunk = file[start_index: end_index]
    inp = char_tensor(lang, chunk[:-1])
    target = char_tensor(lang, chunk[1:])
    # print chunk
    return inp, target

def random_training_set():
    line_size = len(lang.lines)
    while 1:
        index = random.randint(0, line_size - 1)
        chunk = lang.lines[index]
        if len(chunk) > 2:
            #print "train: " , chunk[:-1]
            #print "target: ", chunk[1:]
            inp = char_tensor(lang, chunk[:-1])
            target = char_tensor(lang, chunk[1:])
            break
    return inp, target, len(chunk) - 1


decoder = LSTM(n_characters, hidden_size, n_characters)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

all_losses =[]
loss_avg = 0

def train(inp, target, chunk_size):
    hidden = decoder.init_hidden()
    decoder_optimizer.zero_grad()
    loss = 0

    for c in range(chunk_size):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0]/chunk_size

def save():
    save_filename =  filename.split("txt")[0] + "lstm_h50w"
    torch.save(decoder, save_filename)
    print "Saved as %s " % save_filename

if __name__ == '__main__':
    print "Lstm Training for %d epoches..." % epochs
    all_losses=[]
    for epoch in range(1, epochs + 1):
        inp, target, chunk_size = random_training_set()
        loss = train(inp, target, chunk_size)
        loss_avg += loss
        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / (epochs + 0.0) * 100, loss))
            all_losses.append(loss_avg/print_every)
            loss_avg = 0
            start_char = lang.index2char[random.randint(0, len(lang.index2char)-1)]
            words = generate(lang, decoder, start_char, 10)
            print "predict words is : " + words


    print "Saving..."
    save()
    plt.figure()
    plt.title("lstm 50w")
    plt.plot(all_losses)
    plt.show()





