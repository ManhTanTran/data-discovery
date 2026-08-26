"""Scale-adaptive late interaction implemented with PyTorch when available."""

from __future__ import annotations

from collections.abc import Sequence
import math


Vector = Sequence[float]


def late_interaction_score(
    query_vectors: Sequence[Vector],
    document_vectors: Sequence[Vector],
    top_k: int = 3,
) -> float:
    """Return sum_i mean(top-k_j cosine(q_i, d_j)).

    The PyTorch path is used in production. A numerically equivalent pure-Python
    fallback keeps unit tests and metadata-only workflows usable before optional
    ML dependencies are installed.
    """
    if not query_vectors or not document_vectors:
        return 0.0
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    try:
        import torch
    except ImportError:
        return _python_late_interaction(query_vectors, document_vectors, top_k)

    query = torch.as_tensor(query_vectors, dtype=torch.float32)
    document = torch.as_tensor(document_vectors, dtype=torch.float32)
    if query.ndim != 2 or document.ndim != 2 or query.shape[1] != document.shape[1]:
        raise ValueError("query and document vectors must be 2-D with the same dimension")
    query = torch.nn.functional.normalize(query, p=2, dim=1)
    document = torch.nn.functional.normalize(document, p=2, dim=1)
    similarities = query @ document.T
    actual_k = min(top_k, int(document.shape[0]))
    return float(similarities.topk(actual_k, dim=1).values.mean(dim=1).sum().item())


def normalized_late_interaction_score(
    query_vectors: Sequence[Vector],
    document_vectors: Sequence[Vector],
    top_k: int = 3,
) -> float:
    """Map average token late-interaction cosine to [0, 1] for score fusion."""
    if not query_vectors:
        return 0.0
    raw = late_interaction_score(query_vectors, document_vectors, top_k)
    average = raw / len(query_vectors)
    return max(0.0, min(1.0, (average + 1.0) / 2.0))


def _python_late_interaction(
    query_vectors: Sequence[Vector],
    document_vectors: Sequence[Vector],
    top_k: int,
) -> float:
    dimensions = {len(vector) for vector in [*query_vectors, *document_vectors]}
    if len(dimensions) != 1:
        raise ValueError("query and document vectors must have the same dimension")
    total = 0.0
    for query in query_vectors:
        similarities = sorted(
            (_cosine(query, document) for document in document_vectors), reverse=True
        )[: min(top_k, len(document_vectors))]
        total += sum(similarities) / len(similarities)
    return total


def _cosine(left: Vector, right: Vector) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return dot / (left_norm * right_norm)

