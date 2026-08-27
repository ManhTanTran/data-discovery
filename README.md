# Query-to-SubData Discovery

Pipeline chọn trước corpus, tài liệu và trang liên quan trước khi full parsing,
chunking và embedding. Thiết kế lấy cảm hứng từ bước LLM-Free Visual Selection
của LightSTAR và mở rộng cho PDF, HTML, DOCX, TXT cùng text preview.

```text
Raw Data → Light Preparation → Light Index → Light Query
         → Top-K SubData → Full Parsing → Chunking
         → Full Embedding → Final Retrieval
```

## Chạy nhanh trên máy local

```bash
python -m pip install -e ".[ml,test]"
python scripts/demo_query_to_subdata.py
pytest -q
```

## Google Colab

Mở [`notebooks/lightstar_subdata_colab.ipynb`](notebooks/lightstar_subdata_colab.ipynb),
chọn GPU runtime và chạy lần lượt từ trên xuống. Notebook mount Google Drive,
tự tìm `vidore_v3_industrial/pdfs`, tạo light index bằng multilingual MiniLM,
chọn Top-K corpus/document/page và lưu manifest vào
`AXIOM_DE-RD/data/output/subdata/vidore_v3_industrial/<timestamp>/` trên Google
Drive. Notebook dừng trước full parsing,
chunking và full embedding. Mỗi lần chạy tạo `documents/` chứa nguyên file được
chọn và `pages/` chứa từng trang PDF được chọn dưới dạng PDF một trang.

Notebook đọc toàn bộ query từ thư mục `queries/`, tạo một SubData riêng cho mỗi
query và ghi `batch_summary.csv` với `selection_ms`, `export_subdata_ms` và
`query_to_subdata_ms`. Các cột `deep_processing_ms`, `qa_ms`, `total_e2e_ms`
được để sẵn cho benchmark pipeline hoàn chỉnh.

Chi tiết kiến trúc và data schema nằm trong
[`docs/query_to_subdata_design.md`](docs/query_to_subdata_design.md).
