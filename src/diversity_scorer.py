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


def precompute_corpus_tokens(corpus_texts: List[str]) -> List[frozenset[str]]:
    """Pre-tokenize an entire corpus once. Call this BEFORE the scoring loop."""
    return [_tokenize(t) for t in corpus_texts]


def lexical_diversity(text: str) -> float:
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def edit_distance(a: str, b: str) -> int:
    # Simple Levenshtein implementation.
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def token_overlap(a: str, b: str) -> float:
    set_a = _tokenize(a)
    set_b = _tokenize(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _token_overlap_sets(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """Jaccard overlap between two pre-computed token sets."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def novelty_score(text: str, corpus_texts: list[str],
                  corpus_token_sets: Optional[List[frozenset[str]]] = None) -> float:
    """1 - max token overlap with any corpus sample.

    If *corpus_token_sets* is provided (from ``precompute_corpus_tokens``),
    skips redundant re-tokenization of every corpus text — same result, much faster.
    """
    if not corpus_texts and not corpus_token_sets:
        return 1.0
    text_tokens = _tokenize(text)
    if corpus_token_sets is not None:
        overlaps = [_token_overlap_sets(text_tokens, ct) for ct in corpus_token_sets]
    else:
        overlaps = [token_overlap(text, c) for c in corpus_texts]
    return 1.0 - max(overlaps)


def score_sample(synthetic: str, parent: str, corpus_texts: list[str],
                 corpus_token_sets: Optional[List[frozenset[str]]] = None) -> Dict[str, float]:
    lex = lexical_diversity(synthetic)
    edit = edit_distance(synthetic, parent)
    overlap = token_overlap(synthetic, parent)
    novelty = novelty_score(synthetic, corpus_texts, corpus_token_sets=corpus_token_sets)
    return {
        "lexical_diversity": lex,
        "edit_distance": edit,
        "token_overlap": overlap,
        "novelty_score": novelty,
    }

