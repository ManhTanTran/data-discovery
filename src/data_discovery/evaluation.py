"""Evaluation metrics for hierarchical SubData selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .contracts import FullProcessingResult, LightManifest, SelectionResult


@dataclass(frozen=True)
class GroundTruth:
    corpus_ids: set[str]
    document_ids: set[str]
    page_ids: set[str]


@dataclass
class EvaluationReport:
    corpus_recall_at_k: float
    document_recall_at_k: float
    page_recall_at_k: float
    candidate_reduction_ratio: float
    parsing_cost_reduction: float
    embedding_cost_reduction: float
    end_to_end_latency_ms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def recall_at_k(ranked_ids: Iterable[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    predicted = set(list(ranked_ids)[:k])
    return len(predicted & relevant_ids) / len(relevant_ids)


def evaluate_selection(
    selection: SelectionResult,
    manifest: LightManifest,
    ground_truth: GroundTruth,
    *,
    full_result: FullProcessingResult | None = None,
    top_k_corpora: int | None = None,
    top_k_documents: int | None = None,
    top_k_pages: int | None = None,
) -> EvaluationReport:
    selected_count = len(selection.pages)
    total_candidates = len({segment.page_id for segment in manifest.segments})
    candidate_reduction = (
        1.0 - selected_count / total_candidates if total_candidates else 0.0
    )
    if full_result is not None:
        latency = full_result.latency_ms.get("end_to_end_including_selection", 0.0)
    else:
        latency = (
            manifest.preparation_latency_ms
            + selection.latency_ms.get("total_selection", 0.0)
        )
    return EvaluationReport(
        corpus_recall_at_k=recall_at_k(
            (item.item_id for item in selection.corpora),
            ground_truth.corpus_ids,
            top_k_corpora or len(selection.corpora),
        ),
        document_recall_at_k=recall_at_k(
            (item.item_id for item in selection.documents),
            ground_truth.document_ids,
            top_k_documents or len(selection.documents),
        ),
        page_recall_at_k=recall_at_k(
            (item.item_id for item in selection.pages),
            ground_truth.page_ids,
            top_k_pages or len(selection.pages),
        ),
        candidate_reduction_ratio=max(0.0, min(1.0, candidate_reduction)),
        parsing_cost_reduction=selection.cost.parsing_cost_reduction,
        embedding_cost_reduction=selection.cost.embedding_cost_reduction,
        end_to_end_latency_ms=latency,
    )


def macro_average(reports: Iterable[EvaluationReport]) -> EvaluationReport:
    values = list(reports)
    if not values:
        return EvaluationReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    fields = EvaluationReport.__dataclass_fields__
    averaged = {
        name: sum(float(getattr(report, name)) for report in values) / len(values)
        for name in fields
    }
    return EvaluationReport(**averaged)

