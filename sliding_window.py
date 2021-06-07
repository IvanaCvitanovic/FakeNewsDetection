# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
import math

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
  
  
