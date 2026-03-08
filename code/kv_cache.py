"""
KV Cache Management for Tree Speculative Decoding

Uses crop-based rollback for tree speculation.
"""

import torch
from typing import List


class ManagedKVCache:
    """KV cache with crop-based rollback for tree speculation."""

    def __init__(self, model):
        self.model = model
        self.past_kv = None
        self.length = 0

    def reset(self):
        self.past_kv = None
        self.length = 0

    def prefill(self, input_ids):
        self.reset()
        out = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
        self.past_kv = out.past_key_values
        self.length = input_ids.shape[1]
        return out.logits[:, -1, :]

    def forward_single(self, token):
        if token.dim() == 1:
            token = token.unsqueeze(0)
        pos = torch.tensor([[self.length]], device=token.device)
        out = self.model(
            input_ids=token, position_ids=pos,
            past_key_values=self.past_kv, use_cache=True, return_dict=True,
        )
        self.past_kv = out.past_key_values
        self.length += 1
        return out.logits[:, -1, :]

    def forward_tree(self, ids, pos, mask):
        self._pre_tree_length = self.length
        out = self.model(
            input_ids=ids, position_ids=pos, attention_mask=mask,
            past_key_values=self.past_kv, use_cache=True, return_dict=True,
        )
        self.past_kv = out.past_key_values
        return out.logits

    def rollback_and_replay(self, accepted_tokens, device):
        """Crop KV cache back to pre-tree state, replay accepted tokens."""
        self.past_kv.crop(self._pre_tree_length)
        self.length = self._pre_tree_length

        logits = None
        for tok in accepted_tokens:
            logits = self.forward_single(
                torch.tensor([[tok]], device=device)
            )
        return logits
