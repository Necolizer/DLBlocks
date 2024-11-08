import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()

        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**((2*i)/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*(i+1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x / math.sqrt(self.d_model)
        x = x + pe[:,:x.size(1)]
        return x

class FFN(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.relu()
    
    def forward(self, x):
        return self.linear_2(self.dropout(self.act(self.linear_1(x))))

class MHSA(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.droput = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def attention(self, q, k, v, mask=None, dropout=None):
        scores = torch.einsum('bhlc,bhkc->bhlk', [q,k]) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = self.softmax(scores)
        if dropout is not None:
            scores = dropout(scores)
        out = torch.einsum('bhlk, bhkc->bhlc', [scores, v])
        return out

    def forward(self, x, mask):
        B, L, C = x.shape

        q = self.q_linear(x).view(B, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.k_linear(x).view(B, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        v = self.v_linear(x).view(B, -1, self.h, self.d_k).permute(0, 2, 1, 3)

        scores = self.attention(q,k,v,mask,self.droput)

        scores = scores.permute(0, 2, 1, 3).view(B, L, C)
        out = self.out(scores)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        # self.norm_1 = NormLayer(d_model)
        # self.norm_2 = NormLayer(d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.atten = MHSA(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        x = x + self.dropout_1(self.atten(self.norm_1(x), mask))
        x = x + self.dropout_2(self.ffn(self.norm_2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout):
        super().__init__()
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        # self.norm = NormLayer(d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x