"""Hierarchical corpus -> document -> page Query-to-SubData router."""

from __future__ import annotations

from collections import defaultdict
import math
import random
import time

from .contracts import (
    CostEstimate,
    DiscoveryConfig,
    PreviewSegment,
    ScoredSelection,
    SelectionResult,
)
from .late_interaction import normalized_late_interaction_score
from .light_index import LightIndex


class QueryRouter:
    def __init__(self, index: LightIndex, config: DiscoveryConfig | None = None) -> None:
        self.index = index
        self.config = config or DiscoveryConfig()

    def select(self, query: str) -> SelectionResult:
        total_started = time.perf_counter()
        latency: dict[str, float] = {}

        started = time.perf_counter()
        query_vector, query_token_vectors = self.index.encode_query(query)
        latency["light_query"] = _elapsed_ms(started)

        started = time.perf_counter()
        corpus_scores = self._rank_level("corpus", query, query_vector, query_token_vectors)
        selected_corpora = self._threshold_top_k_with_exploration(
            corpus_scores, self.config.top_k_corpora, "corpus"
        )
        latency["corpus_routing"] = _elapsed_ms(started)

        started = time.perf_counter()
        selected_corpus_ids = {item.item_id for item in selected_corpora}
        allowed_documents = {
            item_id
            for item_id in self.index.level_ids.get("document", [])
            if self.index.items[item_id].corpus_id in selected_corpus_ids
        }
        document_scores = self._rank_level(
            "document",
            query,
            query_vector,
            query_token_vectors,
            allowed_ids=allowed_documents,
        )
        selected_documents = self._threshold_top_k_with_exploration(
            document_scores, self.config.top_k_documents, "document"
        )
        latency["document_selection"] = _elapsed_ms(started)

        started = time.perf_counter()
        selected_document_ids = {item.item_id for item in selected_documents}
        allowed_pages = {
            item_id
            for item_id in self.index.level_ids.get("page", [])
            if self.index.items[item_id].document_id in selected_document_ids
        }
        page_scores = self._rank_level(
            "page",
            query,
            query_vector,
            query_token_vectors,
            allowed_ids=allowed_pages,
        )
        selected_pages = self._threshold_top_k_with_exploration(
            page_scores, self.config.top_k_pages, "page"
        )
        latency["page_selection"] = _elapsed_ms(started)

        plan = self._processing_plan(selected_pages)
        cost = self._cost_estimate(selected_pages)
        latency["total_selection"] = _elapsed_ms(total_started)
        return SelectionResult(
            query=query,
            corpora=selected_corpora,
            documents=selected_documents,
            pages=selected_pages,
            processing_plan=plan,
            latency_ms=latency,
            eliminated={
                "corpora": max(0, len(self.index.level_ids.get("corpus", [])) - len(selected_corpora)),
                "documents": max(0, len(self.index.level_ids.get("document", [])) - len(selected_documents)),
                "pages": max(0, len(self.index.level_ids.get("page", [])) - len(selected_pages)),
            },
            cost=cost,
        )

    def _rank_level(
        self,
        level: str,
        query: str,
        query_vector: list[float],
        query_token_vectors: list[list[float]],
        *,
        allowed_ids: set[str] | None = None,
    ) -> list[ScoredSelection]:
        universe = (
            set(self.index.level_ids.get(level, []))
            if allowed_ids is None
            else set(allowed_ids)
        )
        probe_k = max(
            1,
            min(
                len(universe),
                self._configured_top_k(level) * self.config.ann_candidate_multiplier,
            ),
        )
        semantic = self.index.semantic_candidates(
            level, query_vector, probe_k, allowed_ids=allowed_ids
        )
        lexical = self.index.lexical_scores(level, query, allowed_ids)
        metadata = self.index.metadata_scores(level, query, allowed_ids)

        output: list[ScoredSelection] = []
        for item_id in universe:
            item = self.index.items[item_id]
            semantic_score = semantic.get(item_id, 0.0)
            if level == "document":
                late_score = normalized_late_interaction_score(
                    query_token_vectors,
                    self.index.document_multi_vectors.get(item_id, []),
                    self.config.late_interaction_top_k,
                )
                semantic_score = 0.35 * semantic_score + 0.65 * late_score
            elif level == "page":
                late_score = normalized_late_interaction_score(
                    query_token_vectors,
                    self.index.page_multi_vectors.get(item_id, []),
                    self.config.late_interaction_top_k,
                )
                semantic_score = 0.25 * semantic_score + 0.75 * late_score

            lexical_score = lexical.get(item_id, 0.0)
            metadata_score = metadata.get(item_id, 0.0)
            final_score = (
                self.config.alpha * lexical_score
                + self.config.beta * semantic_score
                + self.config.gamma * metadata_score
            )
            output.append(
                ScoredSelection(
                    item_id=item_id,
                    level=level,
                    corpus_id=item.corpus_id,
                    document_id=item.document_id,
                    page_id=item.page_id,
                    score=final_score,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    metadata_score=metadata_score,
                )
            )
        return sorted(output, key=lambda item: (-item.score, item.item_id))

    def _threshold_top_k_with_exploration(
        self,
        ranked: list[ScoredSelection],
        top_k: int,
        level: str,
    ) -> list[ScoredSelection]:
        selected = [
            item for item in ranked if item.score >= self.config.selection_threshold
        ][:top_k]
        selected_ids = {item.item_id for item in selected}
        rejected = [item for item in ranked if item.item_id not in selected_ids]
        if rejected and self.config.exploration_rate > 0:
            exploration_count = max(1, math.ceil(top_k * self.config.exploration_rate))
            exploration_count = min(exploration_count, len(rejected))
            seed = f"{self.config.random_seed}:{level}:{len(ranked)}"
            rng = random.Random(seed)
            for item in rng.sample(rejected, exploration_count):
                item.exploration = True
                selected.append(item)
        return sorted(
            selected,
            key=lambda item: (item.exploration, -item.score, item.item_id),
        )

    def _configured_top_k(self, level: str) -> int:
        return {
            "corpus": self.config.top_k_corpora,
            "document": self.config.top_k_documents,
            "page": self.config.top_k_pages,
        }[level]

    def _processing_plan(
        self, selected_pages: list[ScoredSelection]
    ) -> list[dict[str, object]]:
        document_map = self.index.manifest.document_map()
        grouped: dict[str, list[ScoredSelection]] = defaultdict(list)
        for item in selected_pages:
            if item.document_id:
                grouped[item.document_id].append(item)

        plan: list[dict[str, object]] = []
        for document_id, pages in grouped.items():
            document = document_map[document_id]
            plan.append(
                {
                    "corpus_id": document.corpus_id,
                    "document_id": document_id,
                    "source_uri": document.uri,
                    "media_type": document.media_type,
                    "page_ids": [item.page_id for item in pages if item.page_id],
                    "page_numbers": sorted(
                        {
                            int(segment.page_number)
                            for segment in self.index.manifest.segments
                            if segment.page_id in {item.page_id for item in pages}
                            and segment.page_number is not None
                        }
                    ),
                    "actions": ["full_parse", "chunk", "full_embed", "final_retrieve"],
                    "contains_exploration": any(item.exploration for item in pages),
                }
            )
        return plan

    def _cost_estimate(self, selected_pages: list[ScoredSelection]) -> CostEstimate:
        manifest = self.index.manifest
        selected_page_ids = {item.page_id for item in selected_pages if item.page_id}
        selected_segments = [
            segment for segment in manifest.segments if segment.page_id in selected_page_ids
        ]
        document_map = manifest.document_map()
        segments_by_document: dict[str, list[PreviewSegment]] = defaultdict(list)
        for segment in selected_segments:
            segments_by_document[segment.document_id].append(segment)
        selected_parse_cost = 0.0
        selected_embedding_tokens = 0
        for document_id, segments in segments_by_document.items():
            document = document_map[document_id]
            if document.metadata.get("extension") == ".pdf":
                selected_parse_cost += sum(
                    segment.estimated_parse_cost for segment in segments
                )
                selected_embedding_tokens += sum(
                    segment.estimated_embedding_tokens for segment in segments
                )
            else:
                # Non-paginated formats are routed by logical preview segment, but
                # full processing opens the selected document once.
                selected_parse_cost += document.estimated_parse_cost
                selected_embedding_tokens += document.estimated_embedding_tokens
        return CostEstimate(
            baseline_parsing_cost=sum(item.estimated_parse_cost for item in manifest.documents),
            selected_parsing_cost=selected_parse_cost,
            baseline_embedding_tokens=sum(
                item.estimated_embedding_tokens for item in manifest.documents
            ),
            selected_embedding_tokens=selected_embedding_tokens,
        )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
