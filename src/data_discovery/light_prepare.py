"""Bounded, LLM-free metadata and preview extraction."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from xml.etree.ElementTree import XMLPullParser
import mimetypes
import re
import time
import zipfile

from .ids import make_id
from .contracts import (
    CorpusRecord,
    DiscoveryConfig,
    LightManifest,
    PreviewSegment,
    SourceDocument,
    SUPPORTED_EXTENSIONS,
)


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


@dataclass
class _Preview:
    title: str
    headings: list[str]
    segments: list[tuple[str, str, int | None]]
    page_count: int | None = None


class LightPreparer:
    """Create light representations without invoking an LLM or full parser."""

    def __init__(
        self,
        config: DiscoveryConfig | None = None,
        *,
        thumbnail_dir: str | Path | None = None,
    ) -> None:
        self.config = config or DiscoveryConfig()
        self.thumbnail_dir = Path(thumbnail_dir) if thumbnail_dir else None

    def prepare(self, corpus_roots: dict[str, str | Path]) -> LightManifest:
        started = time.perf_counter()
        documents: list[SourceDocument] = []
        segments: list[PreviewSegment] = []
        corpora: list[CorpusRecord] = []

        for corpus_id, root_value in corpus_roots.items():
            root = Path(root_value)
            files = list(self._supported_files(root))
            corpus_documents: list[SourceDocument] = []
            corpus_segments: list[PreviewSegment] = []
            for path in files:
                document, previews = self._prepare_document(corpus_id, path)
                documents.append(document)
                segments.extend(previews)
                corpus_documents.append(document)
                corpus_segments.extend(previews)

            summary_parts = [document.title for document in corpus_documents]
            summary_parts.extend(segment.text for segment in corpus_segments[:8])
            corpora.append(
                CorpusRecord(
                    corpus_id=corpus_id,
                    title=root.name or corpus_id,
                    document_ids=[document.document_id for document in corpus_documents],
                    summary=_bounded_join(summary_parts, self.config.max_preview_chars * 2),
                    metadata={"root": str(root), "document_count": len(corpus_documents)},
                    estimated_parse_cost=sum(
                        document.estimated_parse_cost for document in corpus_documents
                    ),
                    estimated_embedding_tokens=sum(
                        document.estimated_embedding_tokens for document in corpus_documents
                    ),
                )
            )

        return LightManifest(
            corpora=corpora,
            documents=documents,
            segments=segments,
            preparation_latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _supported_files(self, root: Path) -> Iterable[Path]:
        candidates = [root] if root.is_file() else root.rglob("*")
        return sorted(
            (
                path
                for path in candidates
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: str(path).casefold(),
        )

    def _prepare_document(
        self, corpus_id: str, path: Path
    ) -> tuple[SourceDocument, list[PreviewSegment]]:
        document_id = make_id(corpus_id, str(path.resolve()))
        extension = path.suffix.lower()
        stat = path.stat()
        if extension == ".pdf":
            preview = self._preview_pdf(path, document_id)
        elif extension in {".html", ".htm"}:
            preview = self._preview_html(path)
        elif extension == ".docx":
            preview = self._preview_docx(path)
        else:
            preview = self._preview_text(path)

        page_count = preview.page_count
        parse_cost = float(page_count or max(1, round(stat.st_size / 1_000_000, 2)))
        embedding_tokens = max(1, stat.st_size // 4)
        media_type = mimetypes.guess_type(path.name)[0] or extension.lstrip(".")
        document = SourceDocument(
            corpus_id=corpus_id,
            document_id=document_id,
            uri=str(path.resolve()),
            media_type=media_type,
            title=preview.title or path.stem,
            byte_size=stat.st_size,
            page_count=page_count,
            headings=preview.headings,
            metadata={"extension": extension, "modified_ns": stat.st_mtime_ns},
            estimated_parse_cost=parse_cost,
            estimated_embedding_tokens=embedding_tokens,
        )

        output: list[PreviewSegment] = []
        segment_token_estimate = max(
            1, embedding_tokens // max(1, len(preview.segments))
        )
        segment_parse_cost = parse_cost / max(1, len(preview.segments))
        for index, (kind, text, page_number) in enumerate(
            preview.segments[: self.config.max_preview_segments_per_document]
        ):
            page_id = (
                make_id(document_id, "page", page_number)
                if page_number is not None
                else make_id(document_id, "segment", index)
            )
            segment_id = make_id(page_id, kind, index)
            thumbnail_uri = self._thumbnail_uri(document_id, page_number)
            output.append(
                PreviewSegment(
                    segment_id=segment_id,
                    corpus_id=corpus_id,
                    document_id=document_id,
                    page_id=page_id,
                    page_number=page_number,
                    text=text[: self.config.max_preview_chars],
                    kind=kind,
                    thumbnail_uri=thumbnail_uri,
                    metadata={"source_uri": str(path.resolve())},
                    estimated_parse_cost=segment_parse_cost,
                    estimated_embedding_tokens=segment_token_estimate,
                )
            )
        if not output:
            page_id = make_id(document_id, "segment", 0)
            output.append(
                PreviewSegment(
                    segment_id=make_id(page_id, "empty"),
                    corpus_id=corpus_id,
                    document_id=document_id,
                    page_id=page_id,
                    text=preview.title or path.stem,
                    kind="metadata",
                    estimated_parse_cost=parse_cost,
                    estimated_embedding_tokens=embedding_tokens,
                )
            )
        return document, output

    def _preview_pdf(self, path: Path, document_id: str) -> _Preview:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF light preparation requires PyMuPDF") from exc

        segments: list[tuple[str, str, int | None]] = []
        headings: list[str] = []
        with fitz.open(path) as pdf:
            metadata = pdf.metadata or {}
            title = str(metadata.get("title") or path.stem).strip()
            limit = min(len(pdf), self.config.max_preview_segments_per_document)
            for page_index in range(limit):
                page = pdf.load_page(page_index)
                text = page.get_text("text", sort=True)[: self.config.max_preview_chars]
                first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
                if first_line:
                    headings.append(first_line[:200])
                segments.append(("pdf_page_text", text or title, page_index))
                if self.config.create_pdf_thumbnails and self.thumbnail_dir:
                    self._render_thumbnail(page, document_id, page_index)
            return _Preview(title, headings, segments, page_count=len(pdf))

    def _render_thumbnail(self, page: object, document_id: str, page_number: int) -> None:
        import fitz

        output_dir = self.thumbnail_dir / document_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"page-{page_number}.png"
        scale = self.config.thumbnail_dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        pixmap.save(str(output_path))

    def _thumbnail_uri(self, document_id: str, page_number: int | None) -> str | None:
        if not self.config.create_pdf_thumbnails or self.thumbnail_dir is None:
            return None
        if page_number is None:
            return None
        path = self.thumbnail_dir / document_id / f"page-{page_number}.png"
        return str(path.resolve()) if path.exists() else None

    def _preview_text(self, path: Path) -> _Preview:
        text = _read_prefix(path, self.config.max_preview_chars * 8)
        headings = [match.group(1).strip() for match in _HEADING_RE.finditer(text)]
        title = headings[0] if headings else path.stem
        segments = _structural_segments(title, headings, text, self.config.max_preview_chars)
        return _Preview(title, headings, segments)

    def _preview_html(self, path: Path) -> _Preview:
        parser = _BoundedHTMLPreview(self.config.max_preview_chars * 8)
        parser.feed(_read_prefix(path, self.config.max_preview_chars * 12))
        parser.close()
        title = parser.title or (parser.headings[0] if parser.headings else path.stem)
        text = "\n".join(parser.parts)
        segments = _structural_segments(
            title, parser.headings, text, self.config.max_preview_chars
        )
        return _Preview(title, parser.headings, segments)

    def _preview_docx(self, path: Path) -> _Preview:
        paragraphs: list[str] = []
        headings: list[str] = []
        title = path.stem
        max_chars = self.config.max_preview_chars * 8
        with zipfile.ZipFile(path) as archive:
            try:
                with archive.open("word/document.xml") as source:
                    parser = XMLPullParser(events=("end",))
                    while sum(len(item) for item in paragraphs) < max_chars:
                        chunk = source.read(8192)
                        if not chunk:
                            break
                        parser.feed(chunk)
                        for _, element in parser.read_events():
                            if _local_name(element.tag) != "p":
                                continue
                            text = "".join(
                                child.text or ""
                                for child in element.iter()
                                if _local_name(child.tag) == "t"
                            ).strip()
                            if text:
                                paragraphs.append(text)
                                style = next(
                                    (
                                        child.attrib.get(
                                            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val",
                                            "",
                                        )
                                        for child in element.iter()
                                        if _local_name(child.tag) == "pStyle"
                                    ),
                                    "",
                                )
                                if str(style).casefold().startswith(("heading", "title")):
                                    headings.append(text)
                            element.clear()
            except KeyError:
                pass
            try:
                core = archive.read("docProps/core.xml")[:16384].decode(
                    "utf-8", errors="replace"
                )
                match = re.search(r"<dc:title[^>]*>(.*?)</dc:title>", core, re.DOTALL)
                if match and match.group(1).strip():
                    title = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            except KeyError:
                pass
        if headings and title == path.stem:
            title = headings[0]
        text = "\n".join(paragraphs)
        return _Preview(
            title,
            headings,
            _structural_segments(title, headings, text, self.config.max_preview_chars),
        )


class _BoundedHTMLPreview(HTMLParser):
    def __init__(self, max_chars: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_chars = max_chars
        self.parts: list[str] = []
        self.headings: list[str] = []
        self.title = ""
        self._capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized == "title" or normalized in {f"h{i}" for i in range(1, 7)}:
            self._capture = normalized

    def handle_endtag(self, tag: str) -> None:
        if self._capture == tag.casefold():
            self._capture = None

    def handle_data(self, data: str) -> None:
        if sum(len(item) for item in self.parts) >= self.max_chars:
            return
        text = " ".join(data.split())
        if not text:
            return
        self.parts.append(text)
        if self._capture == "title":
            self.title = f"{self.title} {text}".strip()
        elif self._capture and self._capture.startswith("h"):
            self.headings.append(text)


def _read_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        return stream.read(max_chars)


def _split_bounded(
    text: str, headings: list[str], max_chars: int
) -> list[tuple[str, str, int | None]]:
    cleaned = text.strip()
    if not cleaned:
        return []
    output: list[tuple[str, str, int | None]] = []
    for offset in range(0, len(cleaned), max_chars):
        piece = cleaned[offset : offset + max_chars].strip()
        if piece:
            kind = "heading_preview" if any(piece.startswith(item) for item in headings) else "text_preview"
            output.append((kind, piece, None))
    return output


def _structural_segments(
    title: str,
    headings: list[str],
    text: str,
    max_chars: int,
) -> list[tuple[str, str, int | None]]:
    output: list[tuple[str, str, int | None]] = []
    if title.strip():
        output.append(("title", title.strip()[:max_chars], None))
    seen = {title.strip().casefold()}
    for heading in headings:
        cleaned = heading.strip()
        if cleaned and cleaned.casefold() not in seen:
            output.append(("heading", cleaned[:max_chars], None))
            seen.add(cleaned.casefold())
    output.extend(_split_bounded(text, headings, max_chars))
    return output


def _bounded_join(parts: Iterable[str], max_chars: int) -> str:
    output: list[str] = []
    length = 0
    for part in parts:
        cleaned = " ".join(str(part).split())
        if not cleaned:
            continue
        remaining = max_chars - length
        if remaining <= 0:
            break
        output.append(cleaned[:remaining])
        length += len(output[-1]) + 1
    return "\n".join(output)


def _local_name(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1]
