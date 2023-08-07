import numpy as np
from typing import List, Any
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

# Architecture adapted from Dallago et al.

class Tokenizer(object):
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    def tokenize(self, seq: str) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in seq])

    def untokenize(self, x) -> str:
        return ''.join([self.t_to_a[t] for t in x])


class ASCollater(object):
    def __init__(self, alphabet: str, tokenizer: object, pad=False, pad_tok=0., backwards=False):
        self.pad = pad
        self.pad_tok = pad_tok
        self.tokenizer = tokenizer
        self.backwards = backwards
        self.alphabet = alphabet

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        sequences = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        sequences = [i.view(-1,1) for i in sequences]
        maxlen = max([i.shape[0] for i in sequences])
        padded = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]),"constant", self.pad_tok) for i in sequences]
        padded = torch.stack(padded)
        mask = [torch.ones(i.shape[0]) for i in sequences]
        mask = [F.pad(i, (0, maxlen - i.shape[0])) for i in mask]
        mask = torch.stack(mask)
        y = data[1]
        y = torch.tensor(y).unsqueeze(-1)
        ohe = []
        for i in padded:
            i_onehot = torch.FloatTensor(maxlen, len(self.alphabet))
            i_onehot.zero_()
            i_onehot.scatter_(1, i, 1)
            ohe.append(i_onehot)
        padded = torch.stack(ohe)
            
        return padded, y, mask

class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                         groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if self.linear:
            x = F.relu(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x


class SeqConvNetwork(nn.Module):
    def __init__(self, num_tokens, max_pool_input_size, kernel_size, dropout):
        super(SeqConvNetwork, self).__init__()
        self.encoder = MaskedConv1d(
            num_tokens, max_pool_input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(
            linear=True, in_dim=max_pool_input_size, out_dim=max_pool_input_size*2)
        self.decoder = nn.Linear(max_pool_input_size*2, 1)
        self.n_tokens = num_tokens
        self.dropout = nn.Dropout(dropout)  
        self.input_size = max_pool_input_size

    def forward(self, x, mask):
        x = F.relu(self.encoder(x, input_mask=mask.repeat(
            self.n_tokens, 1, 1).permute(1, 2, 0)))
        x = x * mask.repeat(self.input_size, 1, 1).permute(1, 2, 0)
        x = self.embedding(x)
        x = self.dropout(x)
        output = self.decoder(x)
        return output

# Feed forward Neural Network with two hidden layers.
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_l1, hidden_size_l2, num_output):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_l1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_l1, hidden_size_l2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_l2, num_output)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
