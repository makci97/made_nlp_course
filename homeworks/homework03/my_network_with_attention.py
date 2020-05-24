import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        """Encodes input sequence

        Args:
            src (torch tensor of shape (t, b)): input sequence
            hidden (torch tensor of shape (n_layers * n_directions, b, h)): prev hidden state (can be None)
        
        Returns:
            outputs (torch tensor of shape (t, b, h)): encoded sequence (dicrections are summed)
            hidden (torch tensor of shape (n_layers * n_directions, b, h)): hidden state
        """
        # embedded = (t, b, embed_size)
        embedded = self.embed(src)## embed input
        
        # output = (t, b, n_directions * h)
        # hidden = (n_layers * n_directions, b, h)
        outputs, hidden = self.gru(embedded, hidden)## forward recurrent unit
        
        # sum bidirectional outputs
        # output = (t, b, n_directions, h)
        outputs = outputs.view(outputs.shape[0], outputs.shape[1], 2, self.hidden_size)
        # output = (t, b, h)
        outputs = outputs.sum(dim=2)
        
        return outputs, hidden
    

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        
        # setup attention parameters
        self.v = nn.Parameter(torch.zeros(hidden_size))
#         self.v = nn.Parameter(torch.zeros(hidden_size * 2))
        
        stdv = 1. / np.sqrt(self.v.shape[0])
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        """Calculates attention weights

        Args:
            hidden (torch tensor of shape (b, h)): prev hidden state (can be None)
            encoder_outputs (torch tensor of shape (t, b, h)): encoded sequence
        
        Returns:
            attn_weights (torch tensor of shape (b, 1, t)): attention weights
        """ 
        
        timestep = encoder_outputs.shape[0]
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)  # [B*T*H]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        
        # [B*T*2H]->[B*T*H]
        ## concat h and encoder_outputs, feed to self.attn and then to softmax 
        energy = F.relu(self.attn(torch.cat([h, encoder_outputs], dim=2))) 
#         energy = self.attn(torch.cat([h, encoder_outputs], dim=2))
        energy = energy.transpose(1, 2)  # [B*H*T]
        
        v = self.v.repeat(encoder_outputs.shape[0], 1).unsqueeze(1)  # [B*1*H]
        attn_weights = torch.bmm(v, energy) ## multiply by v vector to get shape [B*1*T]
        attn_weights = F.softmax(attn_weights.squeeze(1), dim=1).unsqueeze(1)
        
        return attn_weights

    
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.2):
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        """Decodes with attention token by token

        Args:
            input (torch tensor of shape (b,)): input token
            last_hidden (torch tensor of shape (1, b, h)): last hidden
            encoder_outputs (torch tensor of shape (t, b, h)): encoded sequence
        
        Returns:
            output (torch tensor of shape (b, vocab_size)): ouput token distribution
            hidden (torch tensor of shape (1, b, h)): hidden state
            attn_weights (torch tensor of shape (b, 1, t)): attention weights
        """
        # get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        
        # calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)  # (B,1,T)
        context = torch.bmm(attn_weights, encoder_outputs.transpose(1, 0))## apply attention weights to encoder_outputs to get shape # (B,1,N) (don't forget to transpose encoder_outputs)
        context = context.transpose(0, 1)  # (1,B,N)
        
        # combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)## forward recurrent unit 
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1)) # (B, output_size)
        output = F.log_softmax(output, dim=1) # (B, output_size)

        return output, hidden, attn_weights

    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        """Sequence-to-sequence inference

        Args:
            src (torch tensor of shape (t, b)): input sequence
        
        Returns:
            outputs (torch tensor of shape (b, vocab_size)): ouput token distribution
        """
        device = src.device
        max_len = trg.shape[0]
        
        batch_size = src.shape[1]
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(device)

        # output = (t, b, h)
        # hidden = (., b, h)
        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        
        output = trg[0,:].to(device)
        
        for t in range(1, max_len):
            # output = (b, output_size)
            # hidden = (1, b, h)
            # attn_weights = (b, 1, t)
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output) ## apply decoder
            outputs[t] = output
            
            top1 = output.max(1)[1]
            teacher_force = random.random() < teacher_forcing_ratio
            output = (trg[t] if teacher_force else top1).to(device)
            # output = (b, )
            
        return outputs