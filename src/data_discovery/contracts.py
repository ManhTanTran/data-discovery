"""Data contracts for lightweight query-to-SubData selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = frozenset({".pdf", ".html", ".htm", ".docx", ".txt", ".md"})


@dataclass(frozen=True)
class DiscoveryConfig:
    top_k_corpora: int = 3
    top_k_documents: int = 10
    top_k_pages: int = 20
    late_interaction_top_k: int = 3
    selection_threshold: float = 0.15
    alpha: float = 0.25
    beta: float = 0.60
    gamma: float = 0.15
    exploration_rate: float = 0.05
    random_seed: int = 17
    max_preview_chars: int = 1200
    max_preview_segments_per_document: int = 64
    thumbnail_dpi: int = 48
    create_pdf_thumbnails: bool = False
    ann_backend: str = "auto"
    ann_candidate_multiplier: int = 8
    chunk_size_words: int = 180
    chunk_overlap_words: int = 30
    final_top_k: int = 5

    def __post_init__(self) -> None:
        positive_ints = (
            "top_k_corpora",
            "top_k_documents",
            "top_k_pages",
            "late_interaction_top_k",
            "max_preview_chars",
            "max_preview_segments_per_document",
            "ann_candidate_multiplier",
            "chunk_size_words",
            "final_top_k",
        )
        for name in positive_ints:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError("selection_threshold must be in [0, 1]")
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be in [0, 1]")
        if self.chunk_overlap_words < 0 or self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be in [0, chunk_size_words)")
        if self.ann_backend not in {"auto", "faiss", "hnsw", "torch", "python"}:
            raise ValueError("ann_backend must be auto, faiss, hnsw, torch, or python")

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "DiscoveryConfig":
        value = value or {}
        known = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in known})


@dataclass
class SourceDocument:
    corpus_id: str
    document_id: str
    uri: str
    media_type: str
    title: str
    byte_size: int = 0
    page_count: int | None = None
    headings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    estimated_parse_cost: float = 0.0
    estimated_embedding_tokens: int = 0

    @property
    def path(self) -> Path:
        return Path(self.uri)


@dataclass
class PreviewSegment:
    segment_id: str
    corpus_id: str
    document_id: str
    text: str
    page_id: str | None = None
    page_number: int | None = None
    kind: str = "preview"
    thumbnail_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    estimated_parse_cost: float = 0.0
    estimated_embedding_tokens: int = 0


@dataclass
class CorpusRecord:
    corpus_id: str
    title: str
    document_ids: list[str]
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)
    estimated_parse_cost: float = 0.0
    estimated_embedding_tokens: int = 0


@dataclass
class LightManifest:
    corpora: list[CorpusRecord]
    documents: list[SourceDocument]
    segments: list[PreviewSegment]
    preparation_latency_ms: float = 0.0

    def document_map(self) -> dict[str, SourceDocument]:
        return {item.document_id: item for item in self.documents}

    def segment_map(self) -> dict[str, PreviewSegment]:
        return {item.segment_id: item for item in self.segments}

    def segments_by_document(self) -> dict[str, list[PreviewSegment]]:
        grouped: dict[str, list[PreviewSegment]] = {}
        for segment in self.segments:
            grouped.setdefault(segment.document_id, []).append(segment)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredSelection:
    item_id: str
    level: str
    corpus_id: str
    document_id: str | None
    page_id: str | None
    score: float
    lexical_score: float
    semantic_score: float
    metadata_score: float
    exploration: bool = False


@dataclass
class CostEstimate:
    baseline_parsing_cost: float
    selected_parsing_cost: float
    baseline_embedding_tokens: int
    selected_embedding_tokens: int

    @property
    def parsing_cost_reduction(self) -> float:
        return _reduction(self.baseline_parsing_cost, self.selected_parsing_cost)

    @property
    def embedding_cost_reduction(self) -> float:
        return _reduction(self.baseline_embedding_tokens, self.selected_embedding_tokens)


@dataclass
class SelectionResult:
    query: str
    corpora: list[ScoredSelection]
    documents: list[ScoredSelection]
    pages: list[ScoredSelection]
    processing_plan: list[dict[str, Any]]
    latency_ms: dict[str, float]
    eliminated: dict[str, int]
    cost: CostEstimate

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cost"]["parsing_cost_reduction"] = self.cost.parsing_cost_reduction
        payload["cost"]["embedding_cost_reduction"] = self.cost.embedding_cost_reduction
        return payload


@dataclass
class ParsedUnit:
    corpus_id: str
    document_id: str
    page_id: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    corpus_id: str
    document_id: str
    page_id: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    chunk_id: str
    document_id: str
    page_id: str | None
    score: float
    text: str


@dataclass
class FullProcessingResult:
    hits: list[RetrievalHit]
    parsed_units: int
    chunks: int
    embedded_chunks: int
    latency_ms: dict[str, float]
    processed_document_ids: list[str]
    processed_page_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reduction(baseline: float, selected: float) -> float:
    if baseline <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - selected / baseline))

