import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head: int, d_model: int):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.head = head
        self.d_q = d_model // head
        self.d_kv = d_model // head

        self.fc_q = nn.Linear(d_model, head * self.d_q)
        self.fc_k = nn.Linear(d_model, head * self.d_kv)
        self.fc_v = nn.Linear(d_model, head * self.d_kv)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attention_mask):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.head, self.d_q).permute(0, 2, 1, 3)  # b_s, head, nq, d_q
        k = self.fc_k(keys).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 3, 1)    # b_s, head, d_kv, nk
        v = self.fc_v(values).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 1, 3)  # b_s, head, nk, d_kv

        att = torch.matmul(q, k) / math.sqrt(self.d_kv)
        
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -1e4)

        att = torch.softmax(att, dim=-1)
        output = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b_s, nq, -1)
        return output
    
class PositionnalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout, max_len=512):
        super(PositionnalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1, max_len, d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, :x.size(1)]
        pe = pe.expand(x.size(0), -1, -1)
        x = x + pe
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForward, self).__init__()
        self.Linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        return self.Linear2(self.dropout(F.relu(self.Linear1(x))))
            
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, head: int, d_ff: int, dropout: float):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = ScaledDotProductAttention(head, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, attention_mask: torch.Tensor):
        residual = src 
        out = self.self_attn(src, src, src, attention_mask)
        out = self.dropout1(out)
        src = self.norm1(residual + out) 
        residual = src 
        out = self.ffn(src)
        out = self.dropout2(out)
        features = self.norm2(residual + out)

        return features
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, head, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, attention_mask)
        return outputs
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, head: int, d_ff: int, dropout: float):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = ScaledDotProductAttention(head, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.enc_dec_attn = ScaledDotProductAttention(head, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, enc_states: torch.Tensor, enc_attention_mask: torch.Tensor, dec_attention_mask: torch.Tensor):
        residual = tgt
        out = self.self_attn(tgt, tgt, tgt, attention_mask=dec_attention_mask)
        out = self.dropout1(out) # Dropout áp dụng lên output của attention
        tgt = self.norm1(residual + out) # Cộng với residual

        # 2. Cross Attention
        residual = tgt
        out = self.enc_dec_attn(tgt, enc_states, enc_states, attention_mask=enc_attention_mask)
        out = self.dropout2(out)
        tgt = self.norm2(residual + out)

        # 3. FFN
        residual = tgt
        out = self.ffn(tgt)
        out = self.dropout3(out)
        tgt = self.norm3(residual + out)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, head, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, inputs: torch.Tensor, enc_states: torch.Tensor, enc_attention_mask: torch.Tensor, dec_attention_mask: torch.Tensor):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, enc_states, enc_attention_mask, dec_attention_mask)
        
        return outputs

def generate_padding_mask(_seq: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    mask = (_seq == pad_value)
    return mask.unsqueeze(1).unsqueeze(1)
    
def generate_sequential_mask(seq_len: int) -> torch.BoolTensor:
    attn_shape = (seq_len, seq_len)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).bool()
    return subsequent_mask.unsqueeze(0).unsqueeze(0)