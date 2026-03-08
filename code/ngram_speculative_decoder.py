"""
N-gram Tree Speculative Decoder

Training-free speculative decoding combining:
- N-gram dictionary lookups for candidate generation
- Tree attention for parallel verification
- Optimized KV cache with crop-based rollback
"""

import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from ngram_dictionary import NGramDictionary
from candidate_tree import CandidateTree
from tree_attention import generate_tree_buffers, verify_candidates
from kv_cache import ManagedKVCache


@dataclass
class DecodingStats:
    total_tokens: int = 0
    total_steps: int = 0
    accepted_counts: dict = field(default_factory=dict)
    fallback_count: int = 0

    def record(self, n, is_fallback=False):
        self.total_tokens += n + 1
        self.total_steps += 1
        self.accepted_counts[n] = self.accepted_counts.get(n, 0) + 1
        if is_fallback:
            self.fallback_count += 1

    @property
    def tokens_per_step(self):
        return self.total_tokens / max(1, self.total_steps)

    @property
    def speculation_rate(self):
        return 1 - (self.fallback_count / max(1, self.total_steps))

    def __repr__(self):
        return (f"Stats(tokens={self.total_tokens}, steps={self.total_steps}, "
                f"tok/step={self.tokens_per_step:.2f}, spec={self.speculation_rate:.0%}, "
                f"accepts={dict(sorted(self.accepted_counts.items()))})")


class NGramSpeculativeDecoder:
    def __init__(self, model, tokenizer, max_n=6, cont_len=4, budget=64, top_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.max_n = max_n
        self.cont_len = cont_len
        self.top_k = top_k
        self.ngram = NGramDictionary(max_n, cont_len)
        self.tree = CandidateTree(budget)
        self.kv = ManagedKVCache(model)

    @torch.no_grad()
    def generate(self, input_ids, max_new=100, verbose=False):
        input_ids = input_ids.to(self.device)
        tokens = input_ids[0].tolist()
        self.ngram = NGramDictionary(self.max_n, self.cont_len)
        self.ngram.build_from_prompt(tokens)

        if verbose:
            print(f"Prompt: {len(tokens)} tokens, Dict: {self.ngram.stats()}")

        logits = self.kv.prefill(input_ids)
        stats = DecodingStats()
        gen = 0
        eos = self.tokenizer.eos_token_id

        while gen < max_new:
            curr_id = torch.argmax(logits, dim=-1).item()

            # Query and filter continuations
            suffix = tokens[-(self.max_n - 1):]
            conts = self.ngram.query(suffix, self.top_k)
            filtered = [c[1:] for c in conts if len(c) > 1 and c[0] == curr_id]

            has_tree = filtered and self.tree.build(filtered).num_nodes() > 0

            if not has_tree:
                # === FALLBACK ===
                tokens.append(curr_id)
                gen += 1
                stats.record(0, is_fallback=True)
                if curr_id == eos:
                    break
                logits = self.kv.forward_single(
                    torch.tensor([[curr_id]], device=self.device)
                )
                self.ngram.update([curr_id], tokens[-(self.max_n + self.cont_len):])
                continue

            # === TREE SPECULATION ===
            buf = generate_tree_buffers(self.tree, self.device)
            tree_tok = buf["tokens"]

            # Combined: [curr_token, tree_tokens...]
            combined = torch.cat(
                [torch.tensor([[curr_id]], device=self.device),
                 tree_tok.unsqueeze(0)], dim=1
            )
            tree_len = combined.shape[1]

            # Position IDs
            pos = (buf["pos"] + len(tokens)).unsqueeze(0)

            # Attention mask
            past_len = self.kv.length
            dtype = next(self.model.parameters()).dtype
            attn = torch.zeros(1, 1, tree_len, past_len + tree_len,
                             device=self.device, dtype=dtype)
            tree_attn = torch.where(buf["mask"].bool(), 0.0, float("-inf")).to(dtype)
            attn[:, :, :, past_len:] = tree_attn

            # Tree forward
            out_logits = self.kv.forward_tree(combined, pos, attn)

            # Verify
            accept_n, accepted = verify_candidates(out_logits, tree_tok, buf["retrieve"])

            new_toks = [curr_id] + accepted

            # Rollback and replay — correct KV cache
            logits = self.kv.rollback_and_replay(new_toks, self.device)

            tokens.extend(new_toks)
            gen += len(new_toks)
            stats.record(accept_n)

            if verbose and stats.total_steps % 10 == 0:
                print(f"Step {stats.total_steps}: {stats.tokens_per_step:.2f} tok/step, "
                      f"gen={gen}/{max_new}")

            if eos in new_toks:
                break

            self.ngram.update(new_toks, tokens[-(self.max_n + self.cont_len):])

        if verbose:
            print(f"\nDone: {stats}")

        return torch.tensor([tokens], device=self.device), stats
