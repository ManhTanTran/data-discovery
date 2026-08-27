"""Batch Query-to-SubData execution and Drive-friendly artifact export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import csv
import hashlib
import json
import re
import shutil
import time

from .contracts import DiscoveryConfig, LightManifest, SelectionResult
from .query_router import QueryRouter


_ID_FIELDS = ("query_id", "qid", "question_id", "id", "_id")
_TEXT_FIELDS = ("query", "query_text", "question", "text")
_SUPPORTED_QUERY_FILES = frozenset({".json", ".jsonl", ".csv", ".tsv", ".parquet"})


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    text: str


@dataclass(frozen=True)
class BatchResult:
    batch_dir: str
    summary_path: str
    query_count: int
    successful_queries: int
    failed_queries: int
    rows: list[dict[str, Any]]


def load_queries(path: str | Path) -> list[QueryRecord]:
    """Đọc và chuẩn hóa toàn bộ query từ một file hoặc thư mục."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Không tìm thấy nguồn query: {source}")
    files = (
        [source]
        if source.is_file()
        else sorted(
            item
            for item in source.rglob("*")
            if item.is_file() and item.suffix.lower() in _SUPPORTED_QUERY_FILES
        )
    )
    records: list[QueryRecord] = []
    seen: set[tuple[str, str]] = set()
    for file_path in files:
        for row_index, row in enumerate(_read_rows(file_path)):
            normalized = _query_from_row(row, file_path, row_index)
            if normalized is None:
                continue
            key = (normalized.query_id, normalized.text)
            if key not in seen:
                records.append(normalized)
                seen.add(key)
    if not records:
        raise ValueError(
            f"Không đọc được query từ {source}. Cần cột query/query_text/question/text."
        )
    return records


def run_query_batch(
    router: QueryRouter,
    manifest: LightManifest,
    queries: Iterable[QueryRecord],
    output_root: str | Path,
    config: DiscoveryConfig,
    *,
    corpus_name: str = "vidore_v3_industrial",
    copy_documents: bool = True,
    extract_pages: bool = True,
    max_queries: int | None = None,
    offline_timing_ms: dict[str, float] | None = None,
) -> BatchResult:
    """Chạy mọi query, xuất một SubData riêng và ghi timing summary."""
    query_list = list(queries)
    if max_queries is not None:
        query_list = query_list[:max_queries]
    batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_dir = Path(output_root) / "subdata" / corpus_name / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    document_map = manifest.document_map()
    page_to_segment = {}
    for segment in manifest.segments:
        page_to_segment.setdefault(segment.page_id, segment)

    rows: list[dict[str, Any]] = []
    used_folders: set[str] = set()
    for ordinal, query in enumerate(query_list, start=1):
        query_started_at = datetime.now().isoformat(timespec="milliseconds")
        total_started = time.perf_counter()
        folder_name = _unique_query_folder(query, ordinal, used_folders)
        query_dir = batch_dir / folder_name
        query_dir.mkdir(parents=True, exist_ok=True)
        try:
            selection_started = time.perf_counter()
            selection = router.select(query.text)
            selection_ms = _elapsed_ms(selection_started)
            export_started = time.perf_counter()
            export_counts = _export_selection(
                query,
                selection,
                manifest,
                query_dir,
                config,
                document_map=document_map,
                page_to_segment=page_to_segment,
                copy_documents=copy_documents,
                extract_pages=extract_pages,
            )
            export_ms = _elapsed_ms(export_started)
            total_ms = _elapsed_ms(total_started)
            row = {
                "query_id": query.query_id,
                "query": query.text,
                "status": "success",
                "query_started_at": query_started_at,
                "selected_corpora": len(selection.corpora),
                "selected_documents": len(selection.documents),
                "selected_pages": len(selection.pages),
                "copied_documents": export_counts["copied_documents"],
                "extracted_pages": export_counts["extracted_pages"],
                "selection_ms": selection_ms,
                "export_subdata_ms": export_ms,
                "query_to_subdata_ms": total_ms,
                "deep_processing_ms": "",
                "qa_ms": "",
                "total_e2e_ms": "",
                "parsing_cost_reduction": selection.cost.parsing_cost_reduction,
                "embedding_cost_reduction": selection.cost.embedding_cost_reduction,
                "query_dir": str(query_dir),
                "error": "",
            }
        except Exception as exc:  # Tiếp tục benchmark các query còn lại.
            row = {
                "query_id": query.query_id,
                "query": query.text,
                "status": "failed",
                "query_started_at": query_started_at,
                "selected_corpora": 0,
                "selected_documents": 0,
                "selected_pages": 0,
                "copied_documents": 0,
                "extracted_pages": 0,
                "selection_ms": "",
                "export_subdata_ms": "",
                "query_to_subdata_ms": _elapsed_ms(total_started),
                "deep_processing_ms": "",
                "qa_ms": "",
                "total_e2e_ms": "",
                "parsing_cost_reduction": "",
                "embedding_cost_reduction": "",
                "query_dir": str(query_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
            (query_dir / "error.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        rows.append(row)
        print(
            f"[{ordinal}/{len(query_list)}] {query.query_id}: {row['status']} - "
            f"{float(row['query_to_subdata_ms']):.1f} ms"
        )

    summary_path = batch_dir / "batch_summary.csv"
    _write_csv(summary_path, rows)
    successful = sum(row["status"] == "success" for row in rows)
    batch_manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "query_count": len(rows),
        "successful_queries": successful,
        "failed_queries": len(rows) - successful,
        "timing_definition": {
            "selection_ms": "Query encoding + corpus/document/page routing",
            "export_subdata_ms": "Copy selected documents + extract selected pages + write metadata",
            "query_to_subdata_ms": "selection_ms + export_subdata_ms + orchestration overhead",
            "total_e2e_ms": "query_to_subdata_ms + deep_processing_ms + qa_ms",
        },
        "offline_timing_ms": offline_timing_ms or {},
        "config": asdict(config),
    }
    (batch_dir / "batch_manifest.json").write_text(
        json.dumps(batch_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return BatchResult(
        batch_dir=str(batch_dir),
        summary_path=str(summary_path),
        query_count=len(rows),
        successful_queries=successful,
        failed_queries=len(rows) - successful,
        rows=rows,
    )


def _export_selection(
    query: QueryRecord,
    selection: SelectionResult,
    manifest: LightManifest,
    query_dir: Path,
    config: DiscoveryConfig,
    *,
    document_map: dict[str, Any],
    page_to_segment: dict[str | None, Any],
    copy_documents: bool,
    extract_pages: bool,
) -> dict[str, int]:
    selected_pages: list[dict[str, Any]] = []
    for item in selection.pages:
        segment = page_to_segment.get(item.page_id)
        document = document_map.get(item.document_id) if item.document_id else None
        selected_pages.append(
            {
                "corpus_id": item.corpus_id,
                "document_id": item.document_id,
                "page_id": item.page_id,
                "page_number_zero_based": segment.page_number if segment else None,
                "page_number_human": (
                    segment.page_number + 1
                    if segment and segment.page_number is not None
                    else None
                ),
                "source_uri": document.uri if document else None,
                "title": document.title if document else None,
                "preview_text": segment.text if segment else None,
                "score": item.score,
                "lexical_score": item.lexical_score,
                "semantic_score": item.semantic_score,
                "metadata_score": item.metadata_score,
                "exploration": item.exploration,
            }
        )

    copied_documents: list[dict[str, Any]] = []
    documents_dir = query_dir / "documents"
    if copy_documents:
        documents_dir.mkdir(parents=True, exist_ok=True)
        document_ids = {item.document_id for item in selection.documents if item.document_id}
        for document_id in sorted(document_ids):
            document = document_map[document_id]
            source_path = Path(document.uri)
            target_path = documents_dir / f"{document_id}_{source_path.name}"
            status = "source_not_found"
            copied_uri = None
            if source_path.exists():
                shutil.copy2(source_path, target_path)
                status = "copied"
                copied_uri = str(target_path)
            copied_documents.append(
                {
                    "document_id": document_id,
                    "source_uri": str(source_path),
                    "copied_uri": copied_uri,
                    "status": status,
                }
            )

    extracted_pages: list[dict[str, Any]] = []
    pages_dir = query_dir / "pages"
    if extract_pages:
        import pymupdf

        pages_dir.mkdir(parents=True, exist_ok=True)
        for page in selected_pages:
            source_path = Path(page["source_uri"]) if page["source_uri"] else None
            page_number = page["page_number_zero_based"]
            status = "unsupported_or_missing"
            extracted_uri = None
            if (
                source_path is not None
                and source_path.exists()
                and source_path.suffix.lower() == ".pdf"
                and page_number is not None
            ):
                target_path = pages_dir / (
                    f"{page['document_id']}_page_{int(page_number) + 1:04d}.pdf"
                )
                with pymupdf.open(source_path) as source_pdf:
                    if 0 <= int(page_number) < len(source_pdf):
                        with pymupdf.open() as page_pdf:
                            page_pdf.insert_pdf(
                                source_pdf,
                                from_page=int(page_number),
                                to_page=int(page_number),
                            )
                            page_pdf.save(target_path)
                        extracted_uri = str(target_path)
                        status = "extracted"
                    else:
                        status = "page_out_of_range"
            page["extracted_page_uri"] = extracted_uri
            page["extraction_status"] = status
            extracted_pages.append(
                {
                    "document_id": page["document_id"],
                    "page_id": page["page_id"],
                    "page_number_zero_based": page_number,
                    "extracted_page_uri": extracted_uri,
                    "status": status,
                }
            )

    payload = {
        "query_id": query.query_id,
        "query": query.text,
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "selected_corpora": [asdict(item) for item in selection.corpora],
        "selected_documents": [asdict(item) for item in selection.documents],
        "selected_pages": selected_pages,
        "processing_plan": selection.processing_plan,
        "latency_ms": selection.latency_ms,
        "eliminated": selection.eliminated,
        "estimated_cost": selection.to_dict()["cost"],
        "copied_documents": copied_documents,
        "extracted_pages": extracted_pages,
        "config": asdict(config),
    }
    (query_dir / "subdata_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(query_dir / "selected_pages.csv", selected_pages)
    (query_dir / "processing_plan.json").write_text(
        json.dumps(selection.processing_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "copied_documents": sum(item["status"] == "copied" for item in copied_documents),
        "extracted_pages": sum(item["status"] == "extracted" for item in extracted_pages),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("queries", "data", "items", "rows"):
                if isinstance(value.get(key), list):
                    return [item for item in value[key] if isinstance(item, dict)]
            if all(isinstance(item, str) for item in value.values()):
                return [{"query_id": key, "query": text} for key, text in value.items()]
            return [value]
        return []
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
            return list(csv.DictReader(source, delimiter=delimiter))
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Đọc Parquet cần pandas/pyarrow") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    return []


def _query_from_row(
    row: dict[str, Any], path: Path, row_index: int
) -> QueryRecord | None:
    text = next(
        (str(row[field]).strip() for field in _TEXT_FIELDS if row.get(field) is not None),
        "",
    )
    if not text:
        return None
    query_id = next(
        (str(row[field]).strip() for field in _ID_FIELDS if row.get(field) is not None),
        "",
    )
    if not query_id:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        query_id = f"{path.stem}_{row_index:05d}_{digest}"
    return QueryRecord(query_id=query_id, text=text)


def _unique_query_folder(
    query: QueryRecord, ordinal: int, used_folders: set[str]
) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", query.query_id).strip("._-")
    normalized = normalized[:80] or f"query_{ordinal:05d}"
    candidate = f"{ordinal:05d}_{normalized}"
    if candidate in used_folders:
        digest = hashlib.sha1(query.text.encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate}_{digest}"
    used_folders.add(candidate)
    return candidate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
