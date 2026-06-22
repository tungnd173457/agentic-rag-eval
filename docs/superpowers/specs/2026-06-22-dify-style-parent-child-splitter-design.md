# Thiết kế: Thay splitter parent-child bằng cách làm kiểu Dify

Ngày: 2026-06-22
Phạm vi: `general-agent/` trong repo EnterpriseRAG-Bench.

## 1. Mục tiêu

Thay **hẳn** splitter parent-child hiện tại (4-stage, heading/table-aware) trong
`general-agent/ingestion/splitters/` bằng một splitter **recursive-character 2 tầng
kiểu Dify** (`FixedRecursiveCharacterTextSplitter`), port sạch — tương đương hành vi,
không bê nguyên các quirk của Dify. Mục đích: đối chiếu khác biệt thuật toán cắt
chunk giữa hai cách trên cùng benchmark.

Kiến trúc parent-child cốt lõi **không đổi** (và vốn đã giống Dify): chỉ **child**
được embed + search; **parent** là đơn vị ngữ cảnh trả về; mọi kích thước đo bằng
**ký tự**. Việc thay thế chỉ động vào *thuật toán cắt*, không động vào embedding,
schema Weaviate, hay retrieval.

## 2. Bối cảnh hiện tại (cái sẽ bị thay)

- `ingestion/splitters/parent_child.py`: 4 stage — cắt theo H1/H2, H3+ gộp vào;
  table atomic; merge min-size; floor-fold tới fixpoint; size-cap balanced packing;
  rescue/consolidate runt. Đo theo ký tự.
- `ingestion/splitters/blocks.py`: parse markdown thành Block (heading/table/para).
- `ingestion/splitters/clean.py`: `clean_markdown` (logic riêng của repo).
- `config/ingestion_config.py` `SplitConfig`: `PC_PARENT_MAX_CHARS=10000`,
  `PC_PARENT_MIN_CHARS=2000`, `PC_CHILD_MIN_CHARS=300`, `PC_CHILD_MAX_CHARS=512`,
  `PC_CHILD_OVERLAP=40`, `LEVEL3_*`.

**Contract phải giữ nguyên** (downstream không sửa): hàm
`split_document(markdown) -> (list[Parent], list[Child])`, với:
- `Parent`: `parent_id`, `position`, `text`, `char_count`, `doc_hash`,
  `child_ids`, `title`.
- `Child`: `child_id`, `parent_id`, `position`, `text`, `char_count`, `doc_hash`.

Consumers: `eval/ingest_gold_docs.py::_chunk_objects` (dùng các field trên +
`title_by_parent`), schema Weaviate (`services/weaviate.py` có field `title`,
`kind`, `parent_id`...), retrieval (`retrieval/tools/_store.py`, `search_kb.py`).

## 3. Thiết kế mới

### 3.1 Core splitter (recursive-character, port sạch từ Dify)

Hàm thuần, đo bằng ký tự:

```
split_text(text, chunk_size, fixed_separator, separators) -> list[str]
```

Logic (tương đương `FixedRecursiveCharacterTextSplitter.split_text` /
`recursive_split_text` của Dify):

1. Tách `text` theo `fixed_separator` (vd `"\n\n"`).
2. Mảnh ≤ `chunk_size` → giữ; mảnh > `chunk_size` → đệ quy theo `separators`:
   chọn separator đầu tiên xuất hiện trong text, theo thứ tự ưu tiên
   `["\n\n", "。", ". ", " ", ""]`; separator `""` ⇒ cắt theo ký tự.
3. Gộp các mảnh nhỏ liền nhau cho tới sát `chunk_size` (merge).
4. **Overlap = 0** (cố định). Không có tham số overlap.

Khác biệt cốt lõi so với hiện tại: **không parse block, không biết
heading/table, không min-size, không runt/balanced-packing**.

Bỏ các quirk lạ của Dify; vẫn strip separator thừa ở đầu chunk nếu cần (ảnh hưởng
nội dung). Viết gọn theo style repo.

### 3.2 Clean (port từ Dify `CleanProcessor.clean`)

Bỏ `clean_markdown` của repo. Thay bằng hàm clean gọn port từ Dify:

- **Default-clean (luôn bật):** `<|`→`<`, `|>`→`>`, xoá control char
  `[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\xEF\xBF\xBE]`, xoá `U+FFFE`.
- **`remove_extra_spaces` (bật):** `\n{3,}` → `\n\n`; `[\t\f\r\x20 …]{2,}` → `" "`.

(Không bật `remove_urls_emails`.) Lưu ý: `"\n\n"` **không** bị đổi thành `"\n"`;
chỉ 3+ newline mới gộp còn `"\n\n"`.

### 3.3 split_document — dựng parent → child

```
split_document(markdown) -> (list[Parent], list[Child])
```

1. `clean(markdown)` (mục 3.2).
2. Tạo **parent** theo `PARENT_MODE`:
   - `paragraph`: `split_text(text, PARENT_MAX_CHARS, PARENT_SEPARATOR, RECURSIVE_SEPARATORS)`
     → nhiều parent.
   - `full-doc`: cả tài liệu = 1 parent (không cắt).
3. Tạo **child** cho mỗi parent:
   `split_text(parent.text, CHILD_MAX_CHARS, CHILD_SEPARATOR, RECURSIVE_SEPARATORS)`.
4. Gán field giữ đúng contract:
   - `Parent`: `parent_id = f"parent_{i+1:010d}"`, `position` (0-based tuần tự),
     `text`, `char_count = len(text)`, `doc_hash = sha256(text)`,
     `child_ids = [...]`, `title = ""`.
   - `Child`: `child_id = f"child_{n:010d}"` (tuần tự toàn document),
     `parent_id`, `position`, `text`, `char_count`, `doc_hash`.

**`title` luôn `""`**: Dify không sinh title từ splitter. Giữ field để không phải
sửa downstream (schema Weaviate, `_chunk_objects`), nhưng để rỗng.

Bỏ các field nội bộ cũ không còn dùng: `blocks`, `parent_stack`, `anchor`,
`is_atomic`.

### 3.4 Config (`config/ingestion_config.py`, tên trung lập — KHÔNG có "DIFY")

`SplitConfig` mới (bỏ `PC_*_MIN`, `PC_CHILD_OVERLAP`, `LEVEL3_*`):

| Field | Default | Ý nghĩa |
|---|---|---|
| `PARENT_MODE` | `"paragraph"` | `"paragraph"` \| `"full-doc"` |
| `PARENT_MAX_CHARS` | `1024` | chunk_size cắt parent |
| `PARENT_SEPARATOR` | `"\n\n"` | fixed_separator tầng parent |
| `CHILD_MAX_CHARS` | `512` | chunk_size cắt child |
| `CHILD_SEPARATOR` | `"\n"` | fixed_separator tầng child |
| `RECURSIVE_SEPARATORS` | `["\n\n","。",". "," ",""]` | separator đệ quy dùng chung |

Validation (giữ kiểu `_validate_thresholds`): chỉ `CHILD_MAX_CHARS ≤ PARENT_MAX_CHARS`.
Không check khoảng `[50, 4000]`. Không có tham số overlap.

`split_text` luôn được gọi với overlap = 0.

## 4. Đơn vị & ranh giới

- `recursive_char.py` (mới): hàm thuần `split_text(...)` + helper clean. Không phụ
  thuộc Weaviate/embedding. Test độc lập.
- `parent_child.py` (viết lại): chỉ còn `split_document(...)` + dataclass
  `Parent`/`Child` (rút gọn field). Gọi `recursive_char.split_text`.
- `config/ingestion_config.py`: `SplitConfig` mới.
- Xoá: code 4-stage trong `parent_child.py`, `blocks.py`, `clean.py` (nếu không còn
  consumer). Kiểm tra `grep` trước khi xoá.

## 5. Testing (TDD — viết test trước)

`recursive_char`:
- Split theo `fixed_separator`; mảnh dài đệ quy đúng thứ tự separator; fallback cắt
  ký tự khi separator `""`.
- Không chunk nào vượt `chunk_size` (trừ trường hợp 1 token/từ dài hơn cap — ghi rõ
  hành vi giống Dify).
- Gộp mảnh nhỏ tới sát `chunk_size`.

`split_document`:
- `full-doc` → đúng 1 parent; `paragraph` → nhiều parent.
- Mỗi child `parent_id` đúng; `parent.child_ids` khớp tập child.
- ID tuần tự (`parent_NNNNNNNNNN`, `child_NNNNNNNNNN`); chạy lại → cùng kết quả
  (idempotent với cùng input).
- `char_count == len(text)`; `title == ""`.
- clean: `\n{3,}→\n\n`, `\s{2,}→space`, `<|→<`, control char bị xoá.

## 6. Tài liệu so sánh

Bổ sung mục vào `docs/dify-parent-child-chunking.md` (hoặc doc mới): đối chiếu
thuật toán cũ (4-stage heading/table-aware, có min/runt/balanced-packing) vs mới
(recursive-char thuần, 2 config độc lập, `parent_mode`), kèm bảng map config
cũ↔mới và nhận định ảnh hưởng kỳ vọng tới recall/precision.

## 7. Ngoài phạm vi (YAGNI)

- Không làm pluggable strategy (giữ song song 2 splitter) — đã chọn "thay hẳn".
- Không đụng embedding, schema Weaviate, retrieval, orchestrator.
- Không port `remove_urls_emails`, `remove_stopwords`.
- Không tự chạy eval trong phạm vi spec này (có thể chạy sau khi implement xong).
