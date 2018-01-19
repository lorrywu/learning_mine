import model
from data import *
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import train
import torch.nn as nn

n_hidden = 128

learning_rate = 0.1 # If you set this too high, it might explode. If too low, it might not learn

n_categories = len(all_categories)
lstm = model.LSTM(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train_lstm(category_tensor, line_tensor):
    optimizer.zero_grad()
    for i in range(line_tensor.size()[0]):
        output = lstm(line_tensor[i])
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.data[0]

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
all_epoch = 500000
print_every = 1000
plot_every = 1000

if __name__ == "__main__":
    print "lstm"
    for epoch in range(1, all_epoch + 1):
        category, line, category_tensor, line_tensor = train.randomTrainingPair()
        output, loss = train_lstm(category_tensor, line_tensor)
        current_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            guess, guess_i = train.categoryFromOutput(output)
            correct = 'yes' if guess == category else 'no (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, (epoch+0.0) / all_epoch * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    torch.save(lstm, 'char-lstm-classification-50w.pt')

    print "all loss: " , all_losses
    plt.figure()
    plt.plot(all_losses)
    plt.show()

