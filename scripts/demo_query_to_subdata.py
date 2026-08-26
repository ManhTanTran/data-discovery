"""Demo offline cho pipeline Query-to-SubData Selection."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_discovery import (
    DiscoveryConfig,
    FullProcessor,
    LightIndex,
    LightPreparer,
    QueryRouter,
    TorchHashEmbedder,
)
from src.data_discovery.evaluation import GroundTruth, evaluate_selection


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = DiscoveryConfig(
        top_k_corpora=1,
        top_k_documents=2,
        top_k_pages=2,
        late_interaction_top_k=2,
        selection_threshold=0.10,
        alpha=0.55,
        beta=0.35,
        gamma=0.10,
        exploration_rate=0.05,
        ann_backend="auto",
        chunk_size_words=40,
        chunk_overlap_words=8,
        final_top_k=3,
    )

    with tempfile.TemporaryDirectory(prefix="subdata-demo-") as temp_name:
        root = Path(temp_name)
        finance = root / "finance"
        hr = root / "hr"
        science = root / "science"
        finance.mkdir()
        hr.mkdir()
        science.mkdir()
        (finance / "doanh_thu_2024.txt").write_text(
            "# Báo cáo doanh thu 2024\n"
            "Doanh thu tại Việt Nam tăng 18 phần trăm. Miền Nam đóng góp lớn nhất.\n"
            "## Theo tỉnh\nHà Nội 120 tỷ, Thành phố Hồ Chí Minh 240 tỷ.",
            encoding="utf-8",
        )
        (finance / "chi_phi.html").write_text(
            "<html><title>Chi phí vận hành</title><h1>Ngân sách</h1>"
            "<p>Chi phí logistics và vận hành năm 2024.</p></html>",
            encoding="utf-8",
        )
        (hr / "nhan_su.txt").write_text(
            "# Báo cáo nhân sự\nTuyển dụng, đào tạo và tỷ lệ nghỉ việc.",
            encoding="utf-8",
        )
        (science / "vat_ly.txt").write_text(
            "# Thí nghiệm vật lý\nĐo quang phổ và năng lượng photon.",
            encoding="utf-8",
        )

        preparer = LightPreparer(config)
        manifest = preparer.prepare(
            {"finance": finance, "hr": hr, "science": science}
        )
        embedder = TorchHashEmbedder(dimension=256)
        index = LightIndex(manifest, embedder, ann_backend=config.ann_backend)
        query = "Doanh thu theo tỉnh tại Việt Nam năm 2024"
        selection = QueryRouter(index, config).select(query)
        full_result = FullProcessor(manifest, embedder, config).process(selection)

        relevant_document = next(
            item for item in manifest.documents if item.uri.endswith("doanh_thu_2024.txt")
        )
        relevant_page = next(
            item.page_id
            for item in manifest.segments
            if item.document_id == relevant_document.document_id
        )
        report = evaluate_selection(
            selection,
            manifest,
            GroundTruth(
                corpus_ids={"finance"},
                document_ids={relevant_document.document_id},
                page_ids={str(relevant_page)},
            ),
            full_result=full_result,
            top_k_corpora=config.top_k_corpora,
            top_k_documents=config.top_k_documents,
            top_k_pages=config.top_k_pages,
        )

        output = {
            "ann_backend": index.backend_used,
            "selection": selection.to_dict(),
            "full_processing": full_result.to_dict(),
            "evaluation": report.to_dict(),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
