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
tự tìm `vidore_v3_industrial/pdfs`, tạo light index bằng multilingual MiniLM và
chỉ full process các page vượt qua candidate selection.

Chi tiết kiến trúc và data schema nằm trong
[`docs/query_to_subdata_design.md`](docs/query_to_subdata_design.md).
