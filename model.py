# -*- coding: utf-8 -*-
import torch
from util.slidding_window import sliding_chunks_matmul_qk, mask_invalid_locations, sliding_chunks_matmul_pv, PositionalEncoding
from torch import nn
from typing import List
import math
import torch.nn.functional as F

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
        if use_pretrained:
          self.embedding = nn.Embedding.from_pretrained(vocab, freeze=freeze_emb)
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
