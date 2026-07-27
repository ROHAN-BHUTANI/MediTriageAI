# src/diversity_scorer.py
"""Diversity Scorer for synthetic samples.
Computes several simple metrics comparing a synthetic text to its parent.
"""

import re
from typing import Dict, List, Optional

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> frozenset[str]:
    """Tokenize text into a frozenset of lowercased word tokens."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


def precompute_corpus_tokens(corpus_texts: List[str]) -> List[tuple[frozenset[str], int]]:
    """Pre-tokenize an entire corpus once. Call this BEFORE the scoring loop."""
    tokens = [_tokenize(t) for t in corpus_texts]
    return [(t, len(t)) for t in tokens]


def lexical_diversity(text: str) -> float:
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def edit_distance(a: str, b: str) -> int:
    # Highly optimized 1D Levenshtein implementation
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    if n == 0:
        return m
        
    curr = list(range(n + 1))
    for i in range(m):
        b_char = b[i]
        prev = curr
        curr = [i + 1] * (n + 1)
        prev_j_minus_1 = prev[0]
        
        for j in range(1, n + 1):
            curr_j_minus_1 = curr[j - 1]
            prev_j = prev[j]
            
            if a[j - 1] == b_char:
                curr[j] = prev_j_minus_1
            else:
                if prev_j < curr_j_minus_1:
                    curr[j] = 1 + (prev_j if prev_j < prev_j_minus_1 else prev_j_minus_1)
                else:
                    curr[j] = 1 + (curr_j_minus_1 if curr_j_minus_1 < prev_j_minus_1 else prev_j_minus_1)
            prev_j_minus_1 = prev_j
            
    return curr[n]


def token_overlap(a: str, b: str) -> float:
    set_a = _tokenize(a)
    set_b = _tokenize(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _token_overlap_sets(set_a: frozenset[str], set_b: frozenset[str], len_b: int = -1) -> float:
    """Jaccard overlap between two pre-computed token sets."""
    if not set_a or not set_b:
        return 0.0
    len_a = len(set_a)
    if len_b == -1:
        len_b = len(set_b)
    inter_len = len(set_a & set_b)
    return inter_len / (len_a + len_b - inter_len)


def novelty_score(text: str, corpus_texts: list[str],
                  corpus_token_sets: Optional[List[frozenset[str]]] = None,
                  corpus_lens: Optional[List[int]] = None) -> float:
    """1 - max token overlap with any corpus sample."""
    if not corpus_texts and not corpus_token_sets:
        return 1.0
    text_tokens = _tokenize(text)
    if corpus_token_sets is not None:
        if corpus_lens is not None:
            max_overlap = 0.0
            text_len = len(text_tokens)
            if text_len == 0:
                return 1.0
            for ct, ct_len in zip(corpus_token_sets, corpus_lens):
                inter_len = len(text_tokens.intersection(ct))
                if inter_len == 0:
                    continue
                overlap = inter_len / (text_len + ct_len - inter_len)
                if overlap > max_overlap:
                    max_overlap = overlap
            return 1.0 - max_overlap
        else:
            overlaps = [_token_overlap_sets(text_tokens, ct) for ct in corpus_token_sets]
            return 1.0 - max(overlaps)
    else:
        overlaps = [token_overlap(text, c) for c in corpus_texts]
        return 1.0 - max(overlaps)


def score_sample(synthetic: str, parent: str, corpus_texts: list[str],
                 corpus_token_sets: Optional[List[frozenset[str]]] = None,
                 corpus_lens: Optional[List[int]] = None) -> Dict[str, float]:
    lex = lexical_diversity(synthetic)
    edit = edit_distance(synthetic, parent)
    overlap = token_overlap(synthetic, parent)
    novelty = novelty_score(synthetic, corpus_texts, corpus_token_sets=corpus_token_sets, corpus_lens=corpus_lens)
    return {
        "lexical_diversity": lex,
        "edit_distance": edit,
        "token_overlap": overlap,
        "novelty_score": novelty,
    }

