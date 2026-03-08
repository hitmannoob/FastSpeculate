"""
Tree Attention Mask Generation and Verification for Speculative Decoding

Generates attention masks where each candidate token attends only to:
1. The current position (root)
2. Its ancestors in the candidate tree
"""

import torch
from typing import List, Tuple
from candidate_tree import CandidateTree


def generate_tree_buffers(tree: CandidateTree, device="cuda"):
    n = tree.num_nodes()
    if n == 0:
        return {
            "mask": torch.ones(1, 1, 1, 1, device=device),
            "pos": torch.zeros(1, dtype=torch.long, device=device),
            "retrieve": torch.zeros(1, 1, dtype=torch.long, device=device),
            "tokens": torch.tensor([], dtype=torch.long, device=device),
        }

    seq = 1 + n
    mask = torch.eye(seq, dtype=torch.float32)
    mask[:, 0] = 1.0
    for node in tree.nodes[1:]:
        for anc in tree.get_ancestors(node.node_idx):
            mask[node.node_idx, anc] = 1.0

    pos = torch.zeros(seq, dtype=torch.long)
    for i, d in enumerate(tree.get_depths()):
        pos[i + 1] = d

    paths = tree.get_paths()
    max_len = max(len(p) for p in paths) if paths else 0
    retrieve = []
    for path in paths:
        indices = [0]
        curr = tree.root
        for tok in path:
            curr = curr.children[tok]
            indices.append(curr.node_idx)
        indices += [-1] * (max_len + 1 - len(indices))
        retrieve.append(indices)

    return {
        "mask": mask.unsqueeze(0).unsqueeze(0).to(device),
        "pos": pos.to(device),
        "retrieve": torch.tensor(retrieve, dtype=torch.long, device=device),
        "tokens": torch.tensor(tree.get_tokens(), dtype=torch.long, device=device),
    }


def verify_candidates(logits, tokens, retrieve):
    """Find longest accepted path."""
    if tokens.shape[0] == 0:
        return 0, []

    preds = torch.argmax(logits[0], dim=-1)
    best_len, best_tokens = 0, []

    for path in retrieve:
        accepted = []
        for i in range(len(path) - 1):
            node_idx = path[i + 1].item()
            if node_idx < 0:
                break
            pred_pos = path[i].item()
            expected = tokens[node_idx - 1].item()
            if preds[pred_pos].item() == expected:
                accepted.append(expected)
            else:
                break
        if len(accepted) > best_len:
            best_len, best_tokens = len(accepted), accepted

    return best_len, best_tokens
