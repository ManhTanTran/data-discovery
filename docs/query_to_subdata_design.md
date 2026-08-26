# Query-to-SubData Selection

## Mục tiêu và ranh giới

Module chọn một tập dữ liệu nhỏ trước pipeline ingestion nặng. Light Preparation chỉ đọc
metadata và một phần giới hạn của nội dung; nó không OCR, không dựng document tree đầy đủ
và không tạo full embedding. Chỉ document/page vượt ngưỡng hoặc được chọn bởi nhánh
exploration mới được full parse, chunk và full embed.

```text
Raw corpora
  -> Light Preparation (metadata + bounded previews)
  -> Light Index (BM25 + single/multi vectors + ANN)
User query
  -> Light Query
  -> corpus routing
  -> document selection
  -> page/segment selection + late interaction
  -> SubData selection result
  -> selective full parsing
  -> chunking
  -> full embedding
  -> final retrieval
```

## Kiến trúc thư mục

```text
src/data_discovery/
  contracts.py             # schema và cấu hình dùng chung
  light_prepare.py         # preview có giới hạn cho PDF/HTML/DOCX/TXT
  light_index.py           # BM25, single/multi vectors, FAISS/HNSW/PyTorch
  late_interaction.py      # token-to-segment top-k mean
  query_router.py          # route corpus -> document -> page
  full_processor.py        # chỉ parse/chunk/embed SubData được chọn
  evaluation.py            # recall, reduction, cost và latency
scripts/demo_query_to_subdata.py
tests/test_data_discovery.py
configs/data_discovery.json
```

## Data schema

`SourceDocument`

- `corpus_id`, `document_id`, `uri`, `media_type`
- metadata nhẹ: title, headings, tags, byte size, page count

`PreviewSegment`

- `segment_id`, `corpus_id`, `document_id`
- `page_id`, `page_number` nếu có
- text preview bị chặn bởi `max_preview_chars`
- metadata và ước tính parse/embed cost

`LightManifest`

- danh sách corpus, document và preview segment
- tổng cost baseline nếu full-process toàn bộ

`LightIndex`

- single vector cho corpus và document
- multi-vector là tập embedding của preview segment thuộc document/page
- lexical index và metadata lookup
- ANN backend: FAISS hoặc HNSW; PyTorch exact-search là fallback offline

`SelectionResult`

- selected corpus/document/page IDs và score thành phần
- cờ `exploration`
- processing plan
- latency từng stage, số item bị loại và estimated cost

## Thuật toán

Với mỗi level, lexical, semantic và metadata score được chuẩn hóa về `[0, 1]`:

```text
hybrid(q, x) = alpha * lexical(q, x)
             + beta  * semantic(q, x)
             + gamma * metadata(q, x)
```

Dấu gạch đầu dòng trong yêu cầu được hiểu là phép cộng có trọng số. Nếu chủ đích là trừ
một penalty, có thể truyền trọng số âm mà không đổi pipeline.

Late interaction dùng query-token embeddings và preview-segment embeddings:

```text
LI(q, d) = sum_i mean(top_k_j(cos(q_i, d_j)))
```

Để score không phụ thuộc số token khi fusion, router chia `LI` cho số query token và ánh
xạ cosine từ `[-1, 1]` về `[0, 1]`. Công thức nguyên bản vẫn được trả bởi module
`late_interaction.py`.

Random exploration lấy 5% từ phần chưa được chọn, dùng seed cấu hình để tái lập thí
nghiệm. Item exploration được gắn cờ và không bị threshold loại lại.

## Pseudocode

```text
PREPARE(corpus_roots):
  for corpus in corpus_roots:
    for supported file in corpus:
      read stat + format metadata
      if PDF:
        read page count and bounded text preview per page
        optionally render low-resolution thumbnail
      if HTML/TXT/DOCX:
        read bounded title/headings/preview segments
      emit SourceDocument + PreviewSegments
  aggregate corpus summaries
  return LightManifest

BUILD_INDEX(manifest):
  embed corpus summaries as single vectors
  embed document summaries as single vectors
  embed preview segments as multi vectors
  build BM25 structures
  build ANN indexes with FAISS/HNSW (or exact fallback)

SELECT(query):
  q_vector = embed(query)
  q_tokens = embed(content_tokens(query))

  corpus_candidates = ANN + BM25 over all corpora
  corpora = HYBRID_RANK(corpus_candidates)[:top_k_corpora]

  document_candidates = ANN + BM25 restricted to corpora
  for document in candidates:
    semantic = combine(single_vector, late_interaction(q_tokens, doc_segments))
  documents = threshold + top_k_documents + 5% exploration

  page_candidates = ANN + BM25 restricted to documents
  for page in candidates:
    semantic = late_interaction(q_tokens, page_segments)
  pages = threshold + top_k_pages + 5% exploration

  return SelectionResult(ids, scores, plan, latency, reduction, estimated_cost)

FULL_PROCESS(selection):
  assert every source belongs to selected document/page IDs
  selectively parse selected PDF pages; parse selected non-PDF documents
  chunk parsed text
  full-embed chunks
  retrieve final top chunks for query
  return hits + actual latency/cost counters

EVALUATE(predictions, ground_truth):
  compute Corpus/Document/Page Recall@K
  compute candidate reduction ratio
  compute parsing and embedding cost reduction
  report end-to-end latency
```

## Ánh xạ với LightSTAR

- Light Preparation + Light Index + hierarchical routing tương ứng với **LLM-Free
  Visual Selection**: corpus-wide filtering dùng biểu diễn nhẹ, ưu tiên recall và không
  gọi LLM.
- Query-token filtering và late interaction tương ứng với content-grounded query embedding
  và scale-adaptive late interaction.
- Corpus routing, hỗ trợ text/HTML/DOCX, hybrid lexical-metadata score, cost-aware plan và
  random exploration là phần mở rộng cho data lake đa corpus.
- Full parsing/chunking/embedding trên SubData đóng vai trò refinement có ngân sách. Bản
  prototype không bắt buộc MLLM; visual refinement có thể được cắm thêm sau cho các trang
  PDF chứa bảng/hình.

## Chạy prototype

```powershell
# Backend học được + ANN production (cài cả FAISS và HNSW để auto fallback)
python -m pip install -e ".[discovery]"

# Demo offline; nếu chưa cài extra, code dùng feature hashing và exact-search thuần Python
python scripts/demo_query_to_subdata.py

# Unit tests của module
python -m unittest tests.test_data_discovery -v
```

Trong production, khởi tạo `SentenceTransformerEmbedder` để dùng model multilingual nhẹ;
demo dùng `TorchHashEmbedder` để chạy tái lập, không tải model qua mạng. `ann_backend="auto"`
ưu tiên FAISS, sau đó HNSW, PyTorch và cuối cùng mới dùng exact-search thuần Python.

### Nguồn Google Drive đã mount

`configs/data_discovery.json` có thể chứa `source.corpus_roots`. Khi dùng Google Drive for
Desktop ở chế độ stream, truyền mapping này trực tiếp vào Light Preparation:

```python
import json
from pathlib import Path

payload = json.loads(Path("configs/data_discovery.json").read_text(encoding="utf-8"))
config = DiscoveryConfig.from_mapping(payload)
manifest = LightPreparer(config).prepare(payload["source"]["corpus_roots"])
```

Windows biểu diễn shared-folder shortcut bằng file `.lnk`; pipeline dùng đường dẫn thật
trong `.shortcut-targets-by-id` để Python có thể duyệt file như một thư mục bình thường.
