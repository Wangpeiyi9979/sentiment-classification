# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules import Embedder
from modules import Encoder
from modules import Attenter
class TextCNN(nn.Module):
    def __init__(self, opt):

        super(TextCNN, self).__init__()
        self.opt = opt
        self.embedder = Embedder(emb_method=self.opt.emb_method, glove_param=self.opt.glove_param, elmo_param=self.opt.elmo_param, use_gpu=self.opt.use_gpu)

        if self.opt.enc_method == 'cnn':
            self.encoder = Encoder(enc_method=self.opt.enc_method, filters_num=self.opt.filters_num, filters=self.opt.filters, f_dim=self.embedder.word_dim)
            feature_dim = len(self.opt.filters) * self.opt.filters_num
        else:
            self.encoder = Encoder(enc_method=self.opt.enc_method, input_size=self.embedder.word_dim, hidden_size=self.opt.hidden_size, bidirectional=self.opt.bidirectional)
            self.Q = nn.Parameter(torch.randn(self.opt.q_num, self.opt.q_dim))
            self.attenter = Attenter(att_method=self.opt.att_method, f_dim=2*self.opt.hidden_size, q_dim=self.opt.q_dim, q_num=self.opt.q_num)
            feature_dim = self.opt.q_num * self.opt.hidden_size * (int(self.opt.bidirectional)+1)

        self.linear = nn.Linear(feature_dim, self.opt.num_labels)
        self.dropout = nn.Dropout(self.opt.dropout)

    def forward(self, x):

<<<<<<< HEAD
        length = [len(i) for i in x]
        length = torch.LongTensor(length)
        if self.opt.use_gpu:
            length = length.cuda()
        x = self.embedder(x)
        if self.opt.use_mask:
            x = self.encoder(x, length)
        else:
            x = self.encoder(x)
        if self.opt.enc_method == 'cnn':
            x = torch.cat(x, 1)
        else:
            if self.opt.use_mask:
                x = self.attenter(x, self.Q, length)
            else:
                x = self.attenter(x, self.Q)
            x = x.view(x.size(0), -1)
=======
        x = self.embedder(x)
        x = self.encoder(x)
        if self.opt.enc_method == 'cnn':
            x = torch.cat(x, 1)
        else:
            x = self.attenter(x, self.Q)
            x = x.view(x.size(0), -1)

>>>>>>> 27cc3e2c0d39f68c237151ea0d0c70841efc25cf
        x = self.dropout(x)
        x = self.linear(x)    # batch_size * num_label
        return x

