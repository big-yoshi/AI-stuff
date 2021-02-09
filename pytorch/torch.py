#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
from keras.utils import to_categorical


# In[2]:


np.random.seed(0)
f = open('data/shakespear.txt','r')
text = f.readlines()
f.close()


# In[3]:


characters = tuple(set(text))
int2char = dict(enumerate(characters))
char2int = {char: index for index, char in int2char.items()}
encoded = np.array([char2int[char] for char in text])


# In[4]:


def get_batches(arr, n_seqs_in_a_batch, n_characters):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    
    batch_size = n_seqs_in_a_batch * n_characters
    n_batches = len(arr)//batch_size
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs_in_a_batch, -1))
    
    for n in range(0, arr.shape[1], n_characters):
        # The features
        x = arr[:, n:n+n_characters]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_characters]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


# In[5]:


class charLSTM(nn.ModuleList):
    def __init__(self, seq_len, vocab_size, hidden_dim, batch_size):
        super(charLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        self.lstm_1 = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_dim)
        self.lstm_2 = nn.LSTMCell(hidden_dim, hidden_size=hidden_dim)
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hc):
        output_seq = torch.empty((self.seq_len, self.batch_size,
                                 self.vocab_size))
        hc_1 , hc_2 = hc, hc
        
        for t in range(self.seq_len):
            hc_1 = self.lstm_1(x[t],hc_1)
            h_1,c_1 = hc_1
            
            hc_2 = self.lstm_2(h_1,hc_2)
            h_2, c_2 = hc_2
            
            output_seq[t] = self.fc(self.dropout(h_2))
        return output_seq.view((self.seq_len * self.batch_size, -1))
    
    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim),
               torch.zeros(self.batch_size, self.hidden_dim))


# In[6]:


net = charLSTM(seq_len=128, vocab_size=len(char2int), hidden_dim=512, batch_size=128)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# In[9]:


# get the validation and the training data
val_idx = int(len(encoded) * (1 - 0.1))
data, val_data = encoded[:val_idx], encoded[val_idx:]

# empty list for the validation losses
val_losses = list()

# empty list for the samples
samples = list()

for epoch in range(10):
    
    # reinit the hidden and cell steates
    hc = net.init_hidden()
    
    for i, (x, y) in enumerate(get_batches(data, 128, 128)):
        
        # get the torch tensors from the one-hot of training data
        # also transpose the axis for the training set and the targets
        x_train = torch.from_numpy(to_categorical(x, num_classes=net.vocab_size).transpose([1, 0, 2]))
        targets = torch.from_numpy(y.T).type(torch.LongTensor)  # tensor of the target
        
        # zero out the gradients
        optimizer.zero_grad()
        
        # get the output sequence from the input and the initial hidden and cell states
        output = net(x_train, hc)
    
        # calculate the loss
        # we need to calculate the loss across all batches, so we have to flat the targets tensor
        loss = criterion(output, targets.contiguous().view(128*128))
        
        # calculate the gradients
        loss.backward()
        
        # update the parameters of the model
        optimizer.step()
    
        # feedback every 10 batches
        if i % 10 == 0: 
            
            # initialize the validation hidden state and cell state
            val_h, val_c = net.init_hidden()
            
            for val_x, val_y in get_batches(val_data, 128, 128):
                
                # prepare the validation inputs and targets
                val_x = torch.from_numpy(to_categorical(val_x).transpose([1, 0, 2]))
                val_y = torch.from_numpy(val_y.T).type(torch.LongTensor).contiguous().view(128*128)
            
                # get the validation output
                val_output = net(val_x, (val_h, val_c))
                
                # get the validation loss
                val_loss = criterion(val_output, val_y)
                
                # append the validation loss
                val_losses.append(val_loss.item())
                
                # sample 256 chars
                samples.append(''.join([int2char[int_] for int_ in net.predict("A", seq_len=1024)]))
                
            print("Epoch: {}, Batch: {}, Train Loss: {:.6f}, Validation Loss: {:.6f}".format(epoch, i, loss.item(), val_loss.item()))


# In[ ]:




