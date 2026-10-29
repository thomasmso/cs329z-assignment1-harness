"""Problem 3 (bm25): Implement BM25 (8 points)"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from cs329z_hw1.tokenizer import tokenize

K1 = 1.5
B = 0.75


class BM25Index:
    # term -> [(doc_id, score)], doc_id ascending
    terms_to_scores_index: dict[str, list[tuple[int, float]]]
    doc_count: int
    avgdl: float

    def __init__(self, docs: list[str]) -> None:
        # One pass over the corpus: term counts per document, and lengths.
        print(f"Building BM25 index for {len(docs)} documents...")

        # count of terms in each document: term -> [(doc_id, term_frequency)]
        term_counts: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        # token count of each document: doc_id -> token count
        doc_token_counts: list[int] = []

        # The id of a document is its index in the docs list
        for doc_id, text in enumerate(docs):
            tokens = tokenize(text)
            doc_token_counts.append(len(tokens))
            for term, count in Counter(
                tokens
            ).items():  # each item is a tuple: (term, term_frequency)
                term_counts[term].append((doc_id, count))

        self.doc_count = len(docs)
        self.avgdl = sum(doc_token_counts) / self.doc_count

        # Now we can compute BM25 scores for each posting
        self.terms_to_scores_index = {}
        for term, doc_id_and_count_tuples in term_counts.items():
            idf_score = _compute_idf_score(self.doc_count, len(doc_id_and_count_tuples))
            self.terms_to_scores_index[term] = [
                (
                    doc_id,
                    _calculate_term_score(
                        idf_score, term_count, doc_token_counts[doc_id], self.avgdl
                    ),
                )
                for doc_id, term_count in doc_id_and_count_tuples
            ]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """The top-k ``(doc_id, score)`` pairs, best first, ties by lower doc_id."""
        scores: dict[int, float] = {}
        for term in dict.fromkeys(tokenize(query)):  # distinct terms, each scored once
            for doc_id, score in self.terms_to_scores_index.get(term, ()):
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # nsmallest on (-score, doc_id) keeps the k best without sorting every match
        return heapq.nsmallest(k, scores.items(), key=_best_first)


def _best_first(pair: tuple[int, float]) -> tuple[float, int]:
    """Sort key: higher score first, then lower doc_id."""
    doc_id, score = pair
    return (-score, doc_id)


def _compute_idf_score(num_docs: int, doc_freq: int) -> float:
    """Lucene's IDF: the +1 inside the log keeps it positive for common terms."""
    return math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)


def _calculate_term_score(idf: float, tf: int, length: int, avgdl: float) -> float:
    """One term's BM25 contribution to one document."""
    length_norm = 1 - B + B * length / avgdl
    return idf * tf * (K1 + 1) / (tf + K1 * length_norm)
