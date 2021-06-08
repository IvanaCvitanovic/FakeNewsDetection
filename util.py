# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
import math
from torchtext.legacy import data, datasets, vocab

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


def pad_to_window_size(input_ids: torch.Tensor, 
                       one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = int(2 * one_sided_window_size)
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
    return input_ids

def save_vocab(vocab, path):
    with open(path, 'w+') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')
            
def read_vocab(path):
    vocab = dict()
    with open(path, 'r') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab