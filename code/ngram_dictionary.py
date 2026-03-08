"""
N-gram Dictionary for Speculative Decoding

Supports multi-order n-grams (2 through max_n) with:
- Fast suffix lookups via hash maps
- Multiple continuations per suffix
- Frequency and recency tracking for scoring
- Incremental updates as tokens are generated
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Continuation:
    tokens: Tuple[int, ...]
    frequency: int = 1
    last_seen: int = 0

    def score(self, step: int) -> float:
        return self.frequency + 0.1 / (1 + step - self.last_seen)


class NGramDictionary:
    def __init__(self, max_n: int = 6, continuation_length: int = 4):
        self.max_n = max_n
        self.continuation_length = continuation_length
        self._store = defaultdict(list)
        self._step = 0

    def build_from_prompt(self, tokens: List[int]):
        self._extract(tokens)

    def update(self, new_tokens: List[int], context: List[int]):
        self._step += len(new_tokens)
        self._extract(context[-(self.max_n + self.continuation_length):])

    def _extract(self, tokens: List[int]):
        for i in range(len(tokens) - self.continuation_length):
            for slen in range(1, min(self.max_n, i + 2)):
                if i + slen + self.continuation_length > len(tokens):
                    break
                suffix = tuple(tokens[i:i + slen])
                cont = tuple(tokens[i + slen:i + slen + self.continuation_length])
                self._add(suffix, cont)

    def _add(self, suffix, cont):
        for c in self._store[suffix]:
            if c.tokens == cont:
                c.frequency += 1
                c.last_seen = self._step
                return
        if len(self._store[suffix]) < 10:
            self._store[suffix].append(Continuation(cont, 1, self._step))

    def query(self, suffix: List[int], top_k: int = 5) -> List[Tuple[int, ...]]:
        results = []
        seen = set()
        for slen in range(min(len(suffix), self.max_n - 1), 0, -1):
            key = tuple(suffix[-slen:])
            for c in self._store.get(key, []):
                if c.tokens not in seen:
                    seen.add(c.tokens)
                    results.append((c.tokens, c.score(self._step) * (1 + 0.5 * slen)))
        results.sort(key=lambda x: -x[1])
        return [t for t, _ in results[:top_k]]

    def stats(self):
        return {"suffixes": len(self._store), "conts": sum(len(v) for v in self._store.values())}
