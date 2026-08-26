from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data_discovery.contracts import DiscoveryConfig
from src.data_discovery.evaluation import GroundTruth, evaluate_selection, recall_at_k
from src.data_discovery.full_processor import FullProcessor
from src.data_discovery.late_interaction import late_interaction_score
from src.data_discovery.light_index import LightIndex, TorchHashEmbedder
from src.data_discovery.light_prepare import LightPreparer
from src.data_discovery.query_router import QueryRouter


class LateInteractionTests(unittest.TestCase):
    def test_sum_of_per_token_top_k_mean(self) -> None:
        query = [[1.0, 0.0], [0.0, 1.0]]
        document = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        self.assertAlmostEqual(late_interaction_score(query, document, top_k=1), 2.0)


class QueryToSubDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.finance = self.root / "finance"
        self.other = self.root / "other"
        self.finance.mkdir()
        self.other.mkdir()
        (self.finance / "sales.txt").write_text(
            "# Doanh thu Việt Nam 2024\nDoanh thu theo tỉnh và sản phẩm.",
            encoding="utf-8",
        )
        (self.finance / "cost.html").write_text(
            "<title>Chi phí</title><h1>Vận hành</h1><p>Ngân sách logistics.</p>",
            encoding="utf-8",
        )
        (self.other / "physics.txt").write_text(
            "# Vật lý lượng tử\nNăng lượng photon và quang phổ.",
            encoding="utf-8",
        )
        self.config = DiscoveryConfig(
            top_k_corpora=1,
            top_k_documents=1,
            top_k_pages=1,
            late_interaction_top_k=1,
            selection_threshold=0.05,
            alpha=0.8,
            beta=0.15,
            gamma=0.05,
            exploration_rate=0.0,
            max_preview_chars=200,
            ann_backend="python",
            chunk_size_words=16,
            chunk_overlap_words=4,
        )
        self.manifest = LightPreparer(self.config).prepare(
            {"finance": self.finance, "other": self.other}
        )
        self.embedder = TorchHashEmbedder(dimension=128)
        self.index = LightIndex(self.manifest, self.embedder, ann_backend="python")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_light_preparation_is_bounded_and_extracts_structure(self) -> None:
        sales = next(item for item in self.manifest.documents if item.title == "Doanh thu Việt Nam 2024")
        self.assertIn("Doanh thu Việt Nam 2024", sales.headings)
        self.assertGreater(len(self.manifest.segments), 0)
        self.assertTrue(any(segment.kind == "title" for segment in self.manifest.segments))
        self.assertTrue(
            all(len(segment.text) <= self.config.max_preview_chars for segment in self.manifest.segments)
        )

    def test_hierarchical_routing_selects_relevant_subdata(self) -> None:
        selection = QueryRouter(self.index, self.config).select(
            "doanh thu theo tỉnh Việt Nam năm 2024"
        )
        selected_document = self.manifest.document_map()[selection.documents[0].item_id]
        self.assertEqual(selection.corpora[0].item_id, "finance")
        self.assertTrue(selected_document.uri.endswith("sales.txt"))
        self.assertEqual(len(selection.processing_plan), 1)
        self.assertGreater(selection.cost.parsing_cost_reduction, 0.0)

    def test_full_processor_never_reads_rejected_document(self) -> None:
        selection = QueryRouter(self.index, self.config).select("doanh thu Việt Nam")
        result = FullProcessor(self.manifest, self.embedder, self.config).process(selection)
        selected_ids = {item.item_id for item in selection.documents}
        self.assertTrue(set(result.processed_document_ids).issubset(selected_ids))
        self.assertGreater(result.embedded_chunks, 0)
        self.assertTrue(all(hit.document_id in selected_ids for hit in result.hits))

    def test_evaluation_metrics(self) -> None:
        selection = QueryRouter(self.index, self.config).select("doanh thu Việt Nam")
        document_id = selection.documents[0].item_id
        page_id = selection.pages[0].item_id
        report = evaluate_selection(
            selection,
            self.manifest,
            GroundTruth({"finance"}, {document_id}, {page_id}),
            top_k_corpora=1,
            top_k_documents=1,
            top_k_pages=1,
        )
        self.assertEqual(report.corpus_recall_at_k, 1.0)
        self.assertEqual(report.document_recall_at_k, 1.0)
        self.assertEqual(report.page_recall_at_k, 1.0)
        self.assertGreaterEqual(report.candidate_reduction_ratio, 0.0)

    def test_random_exploration_marks_fallback_candidate(self) -> None:
        config = DiscoveryConfig(
            top_k_corpora=1,
            top_k_documents=1,
            top_k_pages=1,
            selection_threshold=0.99,
            exploration_rate=0.05,
            ann_backend="python",
        )
        selection = QueryRouter(self.index, config).select("truy vấn không tồn tại")
        self.assertTrue(any(item.exploration for item in selection.corpora))


class MetricTests(unittest.TestCase):
    def test_recall_at_k(self) -> None:
        self.assertEqual(recall_at_k(["a", "b", "c"], {"b", "x"}, 2), 0.5)


if __name__ == "__main__":
    unittest.main()
