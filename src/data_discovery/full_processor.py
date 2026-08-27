"""Selective full parsing, chunking, embedding, and final retrieval."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any
import math
import re
import time

from .ids import make_id
from .contracts import (
    Chunk,
    DiscoveryConfig,
    FullProcessingResult,
    LightManifest,
    ParsedUnit,
    RetrievalHit,
    SelectionResult,
)
from .light_index import LightEmbedder


class FullProcessor:
    """Run expensive stages exclusively on the selected processing plan."""

    def __init__(
        self,
        manifest: LightManifest,
        embedder: LightEmbedder,
        config: DiscoveryConfig | None = None,
    ) -> None:
        self.manifest = manifest
        self.embedder = embedder
        self.config = config or DiscoveryConfig()

    def process(self, selection: SelectionResult) -> FullProcessingResult:
        total_started = time.perf_counter()
        latency: dict[str, float] = {}
        document_map = self.manifest.document_map()
        selected_document_ids = {
            str(item["document_id"]) for item in selection.processing_plan
        }

        started = time.perf_counter()
        parsed: list[ParsedUnit] = []
        for plan in selection.processing_plan:
            document_id = str(plan["document_id"])
            if document_id not in selected_document_ids:
                raise AssertionError("Attempted to parse an unselected document")
            document = document_map[document_id]
            parsed.extend(
                self._parse_document(
                    document.corpus_id,
                    document_id,
                    Path(document.uri),
                    selected_page_ids=set(str(value) for value in plan.get("page_ids", [])),
                    selected_page_numbers=set(int(value) for value in plan.get("page_numbers", [])),
                )
            )
        latency["full_parsing"] = _elapsed_ms(started)

        started = time.perf_counter()
        chunks = self._chunk(parsed)
        latency["chunking"] = _elapsed_ms(started)

        started = time.perf_counter()
        chunk_vectors = self.embedder.encode([chunk.text for chunk in chunks])
        query_vector = self.embedder.encode([selection.query])[0] if chunks else []
        latency["full_embedding"] = _elapsed_ms(started)

        started = time.perf_counter()
        scored = [
            (chunk, _cosine(query_vector, vector))
            for chunk, vector in zip(chunks, chunk_vectors)
        ]
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        hits = [
            RetrievalHit(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page_id=chunk.page_id,
                score=score,
                text=chunk.text,
            )
            for chunk, score in scored[: self.config.final_top_k]
        ]
        latency["final_retrieval"] = _elapsed_ms(started)
        latency["total_full_processing"] = _elapsed_ms(total_started)
        latency["end_to_end_including_selection"] = (
            latency["total_full_processing"]
            + selection.latency_ms.get("total_selection", 0.0)
            + self.manifest.preparation_latency_ms
        )
        return FullProcessingResult(
            hits=hits,
            parsed_units=len(parsed),
            chunks=len(chunks),
            embedded_chunks=len(chunk_vectors),
            latency_ms=latency,
            processed_document_ids=sorted({item.document_id for item in parsed}),
            processed_page_ids=sorted(
                {item.page_id for item in parsed if item.page_id is not None}
            ),
        )

    def _parse_document(
        self,
        corpus_id: str,
        document_id: str,
        path: Path,
        *,
        selected_page_ids: set[str],
        selected_page_numbers: set[int],
    ) -> list[ParsedUnit]:
        extension = path.suffix.lower()
        if extension == ".pdf":
            return self._parse_pdf(
                corpus_id,
                document_id,
                path,
                selected_page_ids,
                selected_page_numbers,
            )
        if extension in {".html", ".htm"}:
            parser = _FullHTMLText()
            parser.feed(path.read_text(encoding="utf-8-sig", errors="replace"))
            parser.close()
            text = "\n".join(parser.parts)
        elif extension == ".docx":
            text = self._parse_docx(path)
        else:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        return [
            ParsedUnit(
                corpus_id=corpus_id,
                document_id=document_id,
                page_id=next(iter(selected_page_ids), None),
                text=text,
                metadata={"source_uri": str(path), "selective_scope": "document"},
            )
        ]

    def _parse_pdf(
        self,
        corpus_id: str,
        document_id: str,
        path: Path,
        selected_page_ids: set[str],
        selected_page_numbers: set[int],
    ) -> list[ParsedUnit]:
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError("PDF full processing requires PyMuPDF") from exc

        page_lookup = {
            segment.page_number: segment.page_id
            for segment in self.manifest.segments
            if segment.document_id == document_id and segment.page_number is not None
        }
        output: list[ParsedUnit] = []
        with pymupdf.open(path) as pdf:
            for page_number in sorted(selected_page_numbers):
                if not 0 <= page_number < len(pdf):
                    continue
                page_id = page_lookup.get(page_number)
                if selected_page_ids and page_id not in selected_page_ids:
                    continue
                page = pdf.load_page(page_number)
                output.append(
                    ParsedUnit(
                        corpus_id=corpus_id,
                        document_id=document_id,
                        page_id=page_id,
                        text=page.get_text("text", sort=True),
                        metadata={
                            "source_uri": str(path),
                            "page_number": page_number,
                            "selective_scope": "page",
                        },
                    )
                )
        return output

    def _parse_docx(self, path: Path) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX full processing requires python-docx") from exc
        document = Document(str(path))
        parts: list[str] = []
        for item in document.iter_inner_content():
            if hasattr(item, "rows"):
                for row in item.rows:
                    parts.append(" | ".join(cell.text.strip() for cell in row.cells))
            else:
                text = str(getattr(item, "text", "")).strip()
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    def _chunk(self, units: list[ParsedUnit]) -> list[Chunk]:
        size = self.config.chunk_size_words
        step = size - self.config.chunk_overlap_words
        output: list[Chunk] = []
        for unit in units:
            words = re.findall(r"\S+", unit.text)
            for offset in range(0, len(words), step):
                piece = words[offset : offset + size]
                if not piece:
                    continue
                output.append(
                    Chunk(
                        chunk_id=make_id(
                            unit.document_id, unit.page_id, offset, " ".join(piece[:8])
                        ),
                        corpus_id=unit.corpus_id,
                        document_id=unit.document_id,
                        page_id=unit.page_id,
                        text=" ".join(piece),
                        metadata={**unit.metadata, "word_offset": offset},
                    )
                )
                if offset + size >= len(words):
                    break
        return output


class _FullHTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        cleaned = " ".join(data.split())
        if cleaned:
            self.parts.append(cleaned)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return dot / (left_norm * right_norm)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
