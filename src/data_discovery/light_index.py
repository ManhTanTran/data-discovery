"""Hybrid light index with single-vector ANN and multi-vector previews."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
import hashlib
import math
import re

from .contracts import LightManifest


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


class LightEmbedder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...

    def encode_tokens(self, text: str) -> list[list[float]]: ...


class TorchHashEmbedder:
    """Tiny deterministic feature-hashing embedder for offline runs.

    It has no learned weights. Tensor normalization uses PyTorch when installed;
    a pure-Python equivalent is retained for bootstrap/test environments.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.model_name = "torch-feature-hash-v1"

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = [self._vector(text) for text in texts]
        try:
            import torch
        except ImportError:
            return [_normalize(vector) for vector in vectors]
        tensor = torch.as_tensor(vectors, dtype=torch.float32)
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
        return tensor.cpu().tolist()

    def encode_tokens(self, text: str) -> list[list[float]]:
        tokens = _content_tokens(text)
        return self.encode(tokens or [text])

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize(text)
        features = [*tokens]
        features.extend(
            f"{left}_{right}" for left, right in zip(tokens, tokens[1:])
        )
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        return vector


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers adapter for a lightweight learned model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self.dimension = 0

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install the discovery extra to use sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in encoded]

    def encode_tokens(self, text: str) -> list[list[float]]:
        tokens = _content_tokens(text)
        return self.encode(tokens or [text])


@dataclass(frozen=True)
class IndexItem:
    item_id: str
    level: str
    text: str
    corpus_id: str
    document_id: str | None = None
    page_id: str | None = None
    metadata_text: str = ""


class LightIndex:
    """Searchable light representations for corpus/document/page routing."""

    def __init__(
        self,
        manifest: LightManifest,
        embedder: LightEmbedder,
        *,
        ann_backend: str = "auto",
    ) -> None:
        self.manifest = manifest
        self.embedder = embedder
        self.items: dict[str, IndexItem] = {}
        self.level_ids: dict[str, list[str]] = defaultdict(list)
        self.single_vectors: dict[str, list[float]] = {}
        self.document_multi_vectors: dict[str, list[list[float]]] = defaultdict(list)
        self.page_multi_vectors: dict[str, list[list[float]]] = defaultdict(list)
        self._bm25: dict[str, _BM25] = {}
        self._ann: dict[str, _ANNIndex] = {}
        self.backend_requested = ann_backend
        self.backend_used: dict[str, str] = {}
        self._build()

    def _build(self) -> None:
        segment_groups = self.manifest.segments_by_document()
        for corpus in self.manifest.corpora:
            self._add_item(
                IndexItem(
                    corpus.corpus_id,
                    "corpus",
                    f"{corpus.title}\n{corpus.summary}",
                    corpus.corpus_id,
                    metadata_text=" ".join(str(value) for value in corpus.metadata.values()),
                )
            )
        for document in self.manifest.documents:
            previews = segment_groups.get(document.document_id, [])[:4]
            text = "\n".join(
                [document.title, *document.headings[:8], *(item.text for item in previews)]
            )
            self._add_item(
                IndexItem(
                    document.document_id,
                    "document",
                    text,
                    document.corpus_id,
                    document_id=document.document_id,
                    metadata_text=(
                        f"{document.title} {document.media_type} "
                        f"{' '.join(document.headings)} {document.uri}"
                    ),
                )
            )
        for segment in self.manifest.segments:
            page_id = segment.page_id or segment.segment_id
            self._add_item(
                IndexItem(
                    page_id,
                    "page",
                    segment.text,
                    segment.corpus_id,
                    document_id=segment.document_id,
                    page_id=page_id,
                    metadata_text=f"{segment.kind} {segment.metadata}",
                ),
                replace=False,
            )

        for level, ids in self.level_ids.items():
            texts = [self.items[item_id].text for item_id in ids]
            vectors = self.embedder.encode(texts)
            for item_id, vector in zip(ids, vectors):
                self.single_vectors[item_id] = vector
            self._bm25[level] = _BM25(
                [(item_id, self.items[item_id].text) for item_id in ids]
            )
            ann = _ANNIndex(self.backend_requested).build(ids, vectors)
            self._ann[level] = ann
            self.backend_used[level] = ann.backend

        segment_vectors = self.embedder.encode([segment.text for segment in self.manifest.segments])
        for segment, vector in zip(self.manifest.segments, segment_vectors):
            self.document_multi_vectors[segment.document_id].append(vector)
            self.page_multi_vectors[segment.page_id or segment.segment_id].append(vector)

    def _add_item(self, item: IndexItem, *, replace: bool = True) -> None:
        if item.item_id in self.items and not replace:
            existing = self.items[item.item_id]
            self.items[item.item_id] = IndexItem(
                item_id=existing.item_id,
                level=existing.level,
                text=f"{existing.text}\n{item.text}",
                corpus_id=existing.corpus_id,
                document_id=existing.document_id,
                page_id=existing.page_id,
                metadata_text=f"{existing.metadata_text} {item.metadata_text}",
            )
            return
        self.items[item.item_id] = item
        self.level_ids[item.level].append(item.item_id)

    def encode_query(self, query: str) -> tuple[list[float], list[list[float]]]:
        return self.embedder.encode([query])[0], self.embedder.encode_tokens(query)

    def semantic_candidates(
        self,
        level: str,
        query_vector: Sequence[float],
        top_k: int,
        *,
        allowed_ids: set[str] | None = None,
    ) -> dict[str, float]:
        if level not in self._ann:
            return {}
        search_k = len(self.level_ids[level]) if allowed_ids is not None else top_k
        hits = self._ann[level].search(query_vector, max(top_k, search_k))
        output = {
            item_id: max(0.0, min(1.0, (score + 1.0) / 2.0))
            for item_id, score in hits
            if allowed_ids is None or item_id in allowed_ids
        }
        return dict(list(output.items())[:top_k])

    def lexical_scores(
        self, level: str, query: str, allowed_ids: set[str] | None = None
    ) -> dict[str, float]:
        raw = self._bm25[level].score(query, allowed_ids)
        return _normalize_scores(raw)

    def metadata_scores(
        self, level: str, query: str, allowed_ids: set[str] | None = None
    ) -> dict[str, float]:
        query_tokens = set(tokenize(query))
        output: dict[str, float] = {}
        for item_id in self.level_ids.get(level, []):
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            item_tokens = set(tokenize(self.items[item_id].metadata_text))
            output[item_id] = (
                len(query_tokens & item_tokens) / len(query_tokens) if query_tokens else 0.0
            )
        return output


class _ANNIndex:
    def __init__(self, requested: str) -> None:
        self.requested = requested
        self.backend = "python"
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []
        self.index: Any = None

    def build(self, ids: list[str], vectors: list[list[float]]) -> "_ANNIndex":
        self.ids = list(ids)
        self.vectors = [_normalize(list(vector)) for vector in vectors]
        if not vectors:
            return self
        choices = (
            [self.requested]
            if self.requested not in {"auto", "python"}
            else (["faiss", "hnsw", "torch"] if self.requested == "auto" else [])
        )
        for choice in choices:
            try:
                if choice == "faiss":
                    import faiss
                    import numpy as np

                    self.index = faiss.IndexFlatIP(len(self.vectors[0]))
                    self.index.add(np.asarray(self.vectors, dtype="float32"))
                    self.backend = "faiss"
                    return self
                if choice == "hnsw":
                    import hnswlib
                    import numpy as np

                    self.index = hnswlib.Index(space="cosine", dim=len(self.vectors[0]))
                    self.index.init_index(max_elements=len(ids), ef_construction=100, M=16)
                    self.index.add_items(np.asarray(self.vectors, dtype="float32"), range(len(ids)))
                    self.index.set_ef(min(max(50, len(ids)), len(ids)))
                    self.backend = "hnsw"
                    return self
                if choice == "torch":
                    import torch

                    self.index = torch.as_tensor(self.vectors, dtype=torch.float32)
                    self.backend = "torch"
                    return self
            except ImportError:
                if self.requested == choice:
                    raise RuntimeError(f"Requested ANN backend {choice!r} is not installed")
        return self

    def search(self, query: Sequence[float], top_k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        top_k = min(max(1, top_k), len(self.ids))
        normalized_query = _normalize(list(query))
        if self.backend == "faiss":
            import numpy as np

            scores, positions = self.index.search(
                np.asarray([normalized_query], dtype="float32"), top_k
            )
            return [
                (self.ids[int(position)], float(score))
                for position, score in zip(positions[0], scores[0])
                if int(position) >= 0
            ]
        if self.backend == "hnsw":
            import numpy as np

            positions, distances = self.index.knn_query(
                np.asarray([normalized_query], dtype="float32"), k=top_k
            )
            return [
                (self.ids[int(position)], 1.0 - float(distance))
                for position, distance in zip(positions[0], distances[0])
            ]
        if self.backend == "torch":
            import torch

            tensor = torch.as_tensor(normalized_query, dtype=torch.float32)
            scores = self.index @ tensor
            values, positions = scores.topk(top_k)
            return [
                (self.ids[int(position)], float(score))
                for position, score in zip(positions.tolist(), values.tolist())
            ]
        scored = [
            (item_id, sum(a * b for a, b in zip(vector, normalized_query)))
            for item_id, vector in zip(self.ids, self.vectors)
        ]
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


class _BM25:
    def __init__(self, records: list[tuple[str, str]]) -> None:
        self.ids = [item_id for item_id, _ in records]
        self.tokens = [tokenize(text) for _, text in records]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))

    def score(self, query: str, allowed_ids: set[str] | None) -> dict[str, float]:
        query_tokens = tokenize(query)
        total = len(self.ids)
        output: dict[str, float] = {}
        for item_id, tokens, length in zip(self.ids, self.tokens, self.lengths):
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            counts = Counter(tokens)
            score = 0.0
            for term in query_tokens:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                inverse = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * length / max(self.average_length, 1.0)
                )
                score += inverse * frequency * 2.2 / denominator
            output[item_id] = score
        return output


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values(), default=0.0)
    if maximum <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / maximum for key, value in scores.items()}


def _content_tokens(text: str) -> list[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "of", "on", "or", "the", "to", "with",
        "các", "của", "cho", "là", "một", "những", "trong", "và", "về",
    }
    return [token for token in tokenize(text) if token not in stopwords and len(token) > 1]

