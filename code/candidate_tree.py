"""
Candidate Tree Construction for Speculative Decoding

Takes multiple continuation sequences from the n-gram dictionary
and merges them into a trie structure with shared prefixes.
Respects a node budget to limit verification cost.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TreeNode:
    token_id: int
    node_idx: int
    parent_idx: int
    depth: int
    children: dict = field(default_factory=dict)


class CandidateTree:
    def __init__(self, budget: int = 64):
        self.budget = budget
        self.reset()

    def reset(self):
        self.root = TreeNode(-1, 0, -1, 0)
        self.nodes = [self.root]
        self._counter = 1

    def build(self, continuations: List[Tuple[int, ...]]):
        self.reset()
        for cont in continuations:
            curr = self.root
            for tok in cont:
                if self._counter > self.budget:
                    break
                if tok in curr.children:
                    curr = curr.children[tok]
                else:
                    node = TreeNode(tok, self._counter, curr.node_idx, curr.depth + 1)
                    self._counter += 1
                    curr.children[tok] = node
                    self.nodes.append(node)
                    curr = node
        return self

    def num_nodes(self): return len(self.nodes) - 1
    def get_tokens(self): return [n.token_id for n in self.nodes[1:]]
    def get_depths(self): return [n.depth for n in self.nodes[1:]]

    def get_ancestors(self, idx: int) -> List[int]:
        anc = []
        while idx > 0:
            anc.append(idx)
            idx = self.nodes[idx].parent_idx
        return anc[::-1]

    def get_paths(self) -> List[List[int]]:
        paths = []
        def dfs(node, path):
            if node.token_id >= 0:
                path = path + [node.token_id]
            if not node.children:
                if path: paths.append(path)
            else:
                for c in node.children.values(): dfs(c, path)
        dfs(self.root, [])
        return paths
