#coding:utf-8

from model import *
from generate import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from util import *
from com.lorry.nlp.utils import get_wordembed

filename = '../data/tangshi.txt'
hidden_size = 128
n_layer = 1
learning_rate  = 0.01
#chunk_len = 7
epochs = 200000
print_every = 500

lang = read_file(filename)
file = lang.text
print file[0:100]
file_len = lang.text_length
n_characters = len(lang.char2index)
n_input_dim = 128
start = time.time()
word_emb_dict = get_wordembed.get_word_emb()
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
            inp =  chunk[:-1]
            target = chunk[1:]
            break
    return inp, target, len(chunk) - 1


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // (print_every * 50)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

decoder = LSTM_1(n_input_dim, hidden_size, n_characters)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

all_losses =[]
loss_avg = 0

def train(inp, target, chunk_size, epoch):
    hidden = decoder.init_hidden()
    decoder_optimizer.zero_grad()
    loss = 0
    #print inp
    for c in range(chunk_size):
        if inp[c] in word_emb_dict.keys():
            listc = word_emb_dict[inp[c]]
            #targetc = word_emb_dict[target[c]]
            intc = Variable(torch.FloatTensor(listc))

            targc = char_tensor(lang, target[c])
            output, hidden = decoder(intc, hidden)
            loss += criterion(output, targc)

    loss.backward()
    decoder_optimizer.step()


    return loss.data[0]/chunk_size

def save():
    save_filename =  filename.split("txt")[0] + "lstm_h20wlr"
    torch.save(decoder, save_filename)
    print "Saved as %s " % save_filename

if __name__ == '__main__':
    print "Lstm use our own embeding Training for %d epoches..." % epochs
    all_losses=[]
    for epoch in range(1, epochs + 1):
        inp, target, chunk_size = random_training_set()
        loss = train(inp, target, chunk_size, epoch)

        loss_avg += loss
        if epoch % print_every == 0:
            adjust_learning_rate(decoder_optimizer, epoch)
            print "learning rate is ", decoder_optimizer.param_groups[0]['lr']

            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / (epochs + 0.0) * 100, loss))
            all_losses.append(loss_avg/print_every)
            loss_avg = 0
            #start_char = lang.index2char[random.randint(0, len(lang.index2char)-1)]
            #words = generate_embed(lang, decoder, start_char, 10)
            #print "predict words is : " + words


    print "Saving..."
    save()
    plt.figure()
    plt.plot(all_losses)
    plt.show()





