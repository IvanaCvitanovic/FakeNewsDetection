# -*- coding: utf-8 -*-
import torch
from torch import nn
from typing import List
import math
import torch.nn.functional as F
from torchtext.legacy import data, datasets, vocab

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

def sliding_chunks_matmul_qk(q: torch.Tensor, k: torch.Tensor, window: int, padding_value: float):
  b, t, h, s = q.size()

  assert t % window == 0
  assert q.size() == k.size()
  chunk_q = q.view(b, t // window, window, h, s)
  chunk_k = k.view(b, t // window, window, h, s)
  chunk_k_expanded = torch.stack((
      F.pad(chunk_k[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0),
      chunk_k,
      F.pad(chunk_k[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.0),
  ), dim=-1)
  diagonal_attn = torch.einsum('bcxhd,bcyhde->bcxhey', (chunk_q, chunk_k_expanded))
  return diagonal_attn.reshape(b, t, h, 3 * window)



def sliding_chunks_matmul_pv(prob: torch.Tensor, v: torch.Tensor, window: int):
  b, t, h, s = v.size()
  chunk_prob = prob.view(b, t // window, window, h, 3, window)
  chunk_v = v.view(b, t // window, window, h, s)
  chunk_v_expanded = torch.stack((
        F.pad(chunk_v[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0),
        chunk_v,
        F.pad(chunk_v[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.0),
    ), dim=-1)
  context = torch.einsum('bcwhpd,bcdhep->bcwhe', (chunk_prob, chunk_v_expanded))
  return context.reshape(b, t, h, s)


def invalid_locations_mask(window: int, device: str):
  affected_seq_len = window
  diagonals_list = []
  for j in range(-window, 1):
    diagonal_mask = torch.zeros(affected_seq_len, device='cpu', dtype=torch.uint8)
    diagonal_mask[:-j] = 1
    diagonals_list.append(diagonal_mask)
  mask = torch.stack(diagonals_list, dim=-1)
  mask = mask[None, :, None, :]
  ending_mask = mask.flip(dims=(1, 3)).bool().to(device)
  return affected_seq_len, mask.bool().to(device), ending_mask


def mask_invalid_locations(input_tensor: torch.Tensor, window: int) -> torch.Tensor:
  affected_seq_len, beginning_mask, ending_mask = invalid_locations_mask(window, input_tensor.device)
  seq_len = input_tensor.size(1)
  beginning_input = input_tensor[:, :affected_seq_len, :, :window+1]
  beginning_mask = beginning_mask[:, :seq_len].expand(beginning_input.size())
  beginning_input.masked_fill_(beginning_mask, -float('inf'))
  ending_input = input_tensor[:, -affected_seq_len:, :, -(window+1):]
  ending_mask = ending_mask[:, -seq_len:].expand(ending_input.size())
  ending_input.masked_fill_(ending_mask, -float('inf'))
  
  


class LongSelfAttention(nn.Module):
  def __init__(self, layer_id, attention_window: List[int] = None,  embedding_size = 100, num_heads = 8):
    super().__init__()

    assert embedding_size % num_heads == 0, f'Embedding dimension ({embedding_size}) should be divisible by nr. of heads ({num_heads})'

    self.embedding_size = embedding_size
    self.num_heads = num_heads
    self.head_dim = int(self.embedding_size / self.num_heads)


    self.query = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
    self.key = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
    self.value = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

    self.concatheads = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

    self.layer_id = layer_id
    self.attention_window = attention_window[self.layer_id]



  def forward(self, x):
    b, t, e = x.size()
    h = self.num_heads

    assert e == self.embedding_size, f'Input embedding dim ({e}) should match layer embedding dim ({self.embedding_size})'
        
    s = e // h

    # x = (t, b, e)
    x = x.transpose(0, 1)
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    q /= math.sqrt(self.head_dim)

    q = q.view(t, b, h, s).transpose(0, 1)
    k = k.view(t, b, h, s).transpose(0, 1)
    
    # attn_weights = (b, t, h, window*2+1)
    attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
    #print(attn_weights.shape)

    mask_invalid_locations(attn_weights, self.attention_window)

    assert list(attn_weights.size())[:3] == [b, t, h]
    assert attn_weights.size(dim=3) in [self.attention_window * 2 + 1, self.attention_window * 3]

    attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32) 

    attn_weights = attn_weights_float.type_as(attn_weights)
    attn_probs = attn_weights_float.type_as(attn_weights)
    v = v.view(t, b, h, s).transpose(0, 1)
    attn = 0
    attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)

    attn = attn.type_as(x)
    assert list(attn.size()) == [b, t, self.num_heads, self.head_dim]
    attn = attn.reshape(b, t, self.embedding_size).contiguous()
    
    #print(attn.shape)
    
   
    return attn


class TransformerBlock(nn.Module):
    
    def __init__(self, layer_id,  embedding_size, num_heads, max_length, ff_hidden_mult = 4, dropout = 0.0, attention_window: List[int] = None):
        super().__init__()
        self.attention = LongSelfAttention(layer_id=layer_id, attention_window=attention_window, embedding_size=embedding_size, num_heads=num_heads)
        
        
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        
        self.ff = nn.Sequential(
            nn.Linear(embedding_size, ff_hidden_mult * embedding_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * embedding_size, embedding_size)
            )
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        attended = self.attention(x)
        #print(attended.shape)
        #print('x:', x.shape)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)


        return x
    


class CTransformer(nn.Module):
    
    def __init__(self, embedding_size, num_heads, depth, max_length, vocab_size, num_classes, vocab, attention_window,  use_pretrained=True, freeze_emb=False, max_pool = True, dropout = 0.25 ):
        """
        creates a Classification transformer

        """
        
        super().__init__()
        
        self.num_tokens = vocab_size
        self.max_pool = max_pool
        self.vocab = vocab
        if use_pretrained:
          self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=freeze_emb)
        else:
          self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim= embedding_size)
        self.positional_encodings = PositionalEncoding(d_model = embedding_size)
        #self.positional_embeddings = nn.Embedding(num_embeddings = max_length, embedding_dim = embedding_size)
        
        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(layer_id=i, embedding_size = embedding_size, num_heads = num_heads, max_length = max_length, dropout = dropout, attention_window=attention_window)
                )
            
        self.tblocks = nn.Sequential(*tblocks)
        
        self.toprobs = nn.Linear(embedding_size, num_classes)
        
        
        
    def forward(self, x):
        
        tokens = self.embedding(x)
        b, t, e = tokens.size()
        
        
        #positions = self.positional_embeddings(torch.arange(t, device = d()))[None, :, :].expand(b, t, e)
        #x = tokens + positions
        x = self.positional_encodings(tokens)
        #print('x, positional_encoding:', x)
        
        x = self.tblocks(x)
        x = x.max(dim = 1)[0] if self.max_pool else x.mean(dim = 1)
        x = self.toprobs(x)
        
        
        return F.log_softmax(x, dim = 1)
