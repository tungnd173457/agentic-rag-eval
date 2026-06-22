# Dify implement Parent-Child Chunking như thế nào

> Phân tích chi tiết cơ chế *parent-child retrieval* (Dify gọi là **hierarchical /
> "Parent-child" chunk mode**) trong codebase `/home/boltbolt/Desktop/dify`.
> Mọi đường dẫn file đều tương đối với thư mục `api/` của Dify trừ khi ghi rõ.

## 1. Ý tưởng tổng thể

Parent-child là chiến lược chunk **hai tầng** nhằm tách rời *đơn vị để tìm kiếm*
khỏi *đơn vị để đưa vào LLM*:

- **Child chunk** (nhỏ): đơn vị duy nhất được **embed + đẩy vào vector store**.
  Nhỏ ⇒ vector đặc trưng, semantic search chính xác.
- **Parent chunk** (lớn): đơn vị **trả về cho LLM**. Lớn ⇒ giàu ngữ cảnh, câu trả
  lời mạch lạc.

Luồng truy vấn: query → match **child** trong vector DB → tra ngược ra **parent
segment** chứa child đó → trả **nội dung parent** (kèm danh sách child đã match
làm metadata). Nhiều child cùng một parent được gộp lại, parent chỉ trả 1 lần,
lấy `max_score` của các child.

So với *paragraph/text mode* thông thường (parent = child = một segment, chính nó
được embed và cũng được trả về), parent-child tách hai vai trò đó ra.

## 2. Các khái niệm & data model

### 2.1 IndexType — chọn processor

`core/rag/index_processor/constant/index_type.py`

```python
class IndexType(StrEnum):
    PARAGRAPH_INDEX = "text_model"
    QA_INDEX = "qa_model"
    PARENT_CHILD_INDEX = "hierarchical_model"
```

`doc_form` của một `Document` (bảng documents) mang một trong ba giá trị này.
`PARENT_CHILD_INDEX` (`"hierarchical_model"`) kích hoạt `ParentChildIndexProcessor`
qua `index_processor_factory.py`.

### 2.2 ParentMode — hai kiểu tạo parent

`services/entities/knowledge_entities/knowledge_entities.py`

```python
class ParentMode(StrEnum):
    FULL_DOC = "full-doc"     # cả tài liệu là MỘT parent
    PARAGRAPH = "paragraph"   # tài liệu được cắt thành NHIỀU parent
```

### 2.3 Rule — cấu hình chunk

```python
class Segmentation(BaseModel):
    separator: str = "\n"
    max_tokens: int          # LƯU Ý: thực chất đếm theo KÝ TỰ (xem §5)
    chunk_overlap: int = 0

class Rule(BaseModel):
    pre_processing_rules: list[PreProcessingRule] | None = None
    segmentation: Segmentation | None = None              # luật cắt PARENT
    parent_mode: Literal["full-doc", "paragraph"] | None = None
    subchunk_segmentation: Segmentation | None = None     # luật cắt CHILD

class ProcessRule(BaseModel):
    mode: Literal["automatic", "custom", "hierarchical"]
    rules: Rule | None = None
```

Điểm mấu chốt: parent-child dùng **hai bộ Segmentation độc lập** — `segmentation`
cho parent và `subchunk_segmentation` cho child.

### 2.4 Document trong tầng RAG (in-memory)

`core/rag/models/document.py`

```python
class ChildDocument(BaseModel):
    page_content: str
    vector: list[float] | None = None
    metadata: dict = Field(default_factory=dict)

class Document(BaseModel):
    page_content: str
    vector: list[float] | None = None
    metadata: dict = Field(default_factory=dict)
    children: list[ChildDocument] | None = None    # parent ôm các child

# Dùng cho luồng workflow / index() — chunk đã có sẵn nội dung parent+child
class ParentChildChunk(BaseModel):
    parent_content: str
    child_contents: list[str]

class ParentChildStructureChunk(BaseModel):
    parent_child_chunks: list[ParentChildChunk]
    parent_mode: str = "paragraph"
```

Quan hệ parent→child trong RAM là quan hệ chứa (`Document.children`).

### 2.5 Persistence model (DB)

`models/dataset.py`

- **`DocumentSegment`** = **parent** được lưu xuống DB (cột `content`,
  `index_node_id`, `index_node_hash`, `position`, `word_count`, `tokens`…).
- **`ChildChunk`** = **child**, bảng riêng `child_chunks`
  (migration `2024_11_22_0701-e19037032219_parent_child_index.py`):

  ```python
  class ChildChunk(Base):
      id, tenant_id, dataset_id, document_id
      segment_id        # FK -> DocumentSegment.id  (link child -> parent)
      position
      content
      word_count
      index_node_id     # = doc_id, KEY để khớp với vector store
      index_node_hash
      type              # "automatic" | "customized"
      ...
  ```

  Index quan trọng: `(index_node_id, dataset_id)` và `segment_id` — để tra ngược
  từ kết quả vector (`index_node_id`) ra parent (`segment_id`) cực nhanh.

- **`DocumentSegment.child_chunks`** (property): chỉ trả child khi
  `process_rule.mode == "hierarchical"` **và** `parent_mode != FULL_DOC`, query
  `ChildChunk` theo `segment_id` order theo `position`.

Tóm tắt liên kết:

```
Dataset ─┬─ DocumentSegment (PARENT, content trả cho LLM)
         │        ▲ segment_id
         └─ ChildChunk (CHILD, content được embed) ── index_node_id ──► Vector Store
```

## 3. Pipeline indexing

Lớp trung tâm: `core/rag/index_processor/processor/parent_child_index_processor.py`
(`ParentChildIndexProcessor`, kế thừa `BaseIndexProcessor`). Vòng đời:
`extract → transform → load` (cho luồng upload file), hoặc `index` (cho luồng
workflow / chunk đã có sẵn).

### 3.1 extract()

```python
def extract(self, extract_setting, **kwargs):
    return ExtractProcessor.extract(
        extract_setting=extract_setting,
        is_automatic=(kwargs.get("process_rule_mode") in ("automatic", "hierarchical")),
    )
```

Chỉ đọc file thô (PDF, docx, md…) ra danh sách `Document`. Chưa cắt chunk.

### 3.2 transform() — trái tim của parent-child

`transform()` đọc `Rule` từ `process_rule` rồi rẽ nhánh theo `parent_mode`.

**Nhánh PARAGRAPH** (cắt tài liệu thành nhiều parent):

```python
splitter = self._get_splitter(                       # splitter cho PARENT
    processing_rule_mode=process_rule.get("mode"),
    max_tokens=rules.segmentation.max_tokens,
    chunk_overlap=rules.segmentation.chunk_overlap,
    separator=rules.segmentation.separator,
    embedding_model_instance=kwargs.get("embedding_model_instance"),
)
for document in documents:
    document.page_content = CleanProcessor.clean(document.page_content, process_rule)
    document_nodes = splitter.split_documents([document])      # -> các PARENT
    for document_node in document_nodes:
        if document_node.page_content.strip():
            document_node.metadata["doc_id"]   = str(uuid.uuid4())
            document_node.metadata["doc_hash"] = helper.generate_text_hash(...)
            # bỏ ký tự separator thừa ở đầu ("." hoặc "。")
            ...
            child_nodes = self._split_child_nodes(document_node, rules, ...)  # -> CHILD
            document_node.children = child_nodes
```

- Mỗi parent được gán `doc_id` (uuid) + `doc_hash` vào metadata.
- Mỗi parent lại được cắt tiếp thành child bằng `_split_child_nodes`.
- Khi `preview=True` chỉ lấy 10 parent đầu để xem trước.

**Nhánh FULL_DOC** (cả tài liệu là một parent duy nhất):

```python
page_content = "\n".join(d.page_content for d in documents)
document = Document(page_content=page_content, metadata=documents[0].metadata)
child_nodes = self._split_child_nodes(document, rules, ...)
# preview giới hạn CHILD_CHUNKS_PREVIEW_NUMBER (mặc định 50)
document.children = child_nodes
document.metadata["doc_id"]  = str(uuid.uuid4())
document.metadata["doc_hash"] = helper.generate_text_hash(...)
all_documents.append(document)
```

Ở FULL_DOC **không có bước cắt parent** — toàn bộ văn bản nối lại làm parent, chỉ
cắt child. Phù hợp tài liệu ngắn cần giữ nguyên ngữ cảnh toàn cục.

### 3.3 _split_child_nodes() — cắt child từ một parent

```python
def _split_child_nodes(self, document_node, rules, process_rule_mode, embedding_model_instance):
    if not rules.subchunk_segmentation:
        raise ValueError("No subchunk segmentation found in rules.")
    child_splitter = self._get_splitter(           # splitter RIÊNG cho child
        processing_rule_mode=process_rule_mode,
        max_tokens=rules.subchunk_segmentation.max_tokens,
        chunk_overlap=rules.subchunk_segmentation.chunk_overlap,
        separator=rules.subchunk_segmentation.separator,
        embedding_model_instance=embedding_model_instance,
    )
    child_documents = child_splitter.split_documents([document_node])
    for child_document_node in child_documents:
        if child_document_node.page_content.strip():
            child = ChildDocument(page_content=..., metadata=document_node.metadata)
            child.metadata["doc_id"]   = str(uuid.uuid4())   # mỗi child 1 uuid riêng
            child.metadata["doc_hash"] = helper.generate_text_hash(...)
            ...
    return child_nodes
```

Mỗi child nhận `doc_id` uuid riêng (sẽ thành `index_node_id` ở DB và là id trong
vector store). Child **kế thừa metadata của parent** (gồm `document_id`,
`dataset_id`…), nhưng có `doc_id`/`doc_hash` của riêng nó.

### 3.4 load() — chỉ embed CHILD

```python
def load(self, dataset, documents, with_keywords=True, **kwargs):
    if dataset.indexing_technique == "high_quality":
        vector = Vector(dataset)
        for document in documents:
            child_documents = document.children
            if child_documents:
                formatted = [Document(**c.model_dump()) for c in child_documents]
                vector.create(formatted)        # CHỈ child được đẩy vào vector DB
```

Đây là đặc trưng quan trọng nhất: **parent KHÔNG được embed**. Chỉ child đi vào
vector store. Parent được lưu ở DB (qua docstore) để tra cứu khi retrieve.

> `indexing_technique == "economy"` (keyword/BM25) sẽ không vào nhánh này — parent
> -child embedding chỉ áp dụng cho `high_quality`.

### 3.5 index() — luồng "chunk đã có sẵn" (workflow / API)

Khi caller đã có sẵn cặp parent/child (ví dụ Knowledge Index node trong workflow,
hoặc API truyền structure), gọi thẳng `index()` thay vì `transform/load`:

```python
def index(self, dataset, document, chunks):
    parent_childs = ParentChildStructureChunk(**chunks)
    documents = []
    for parent_child in parent_childs.parent_child_chunks:
        metadata = {... "doc_id": uuid4(), "doc_hash": hash(parent_content)}
        child_documents = [ChildDocument(page_content=child, metadata={...uuid4()...})
                           for child in parent_child.child_contents]
        documents.append(Document(page_content=parent_child.parent_content,
                                   metadata=metadata, children=child_documents))
    # ghi lại process rule = hierarchical + parent_mode
    db.session.add(DatasetProcessRule(mode="hierarchical",
                   rules=json.dumps({"parent_mode": parent_childs.parent_mode}), ...))
    # lưu parent -> DocumentSegment, child -> ChildChunk
    doc_store.add_documents(docs=documents, save_child=True)
    # embed child
    if dataset.indexing_technique == "high_quality":
        vector.create(all_child_documents)
```

### 3.6 Lưu xuống DB — DatasetDocumentStore.add_documents(save_child=True)

`core/rag/docstore/dataset_docstore.py`

```python
segment_document = DocumentSegment(            # PARENT
    index_node_id=doc.metadata["doc_id"],
    index_node_hash=doc.metadata["doc_hash"],
    content=doc.page_content,
    word_count=len(doc.page_content),
    position=max_position, enabled=False, ...
)
db.session.add(segment_document); db.session.flush()
if save_child and doc.children:
    for position, child in enumerate(doc.children, start=1):
        db.session.add(ChildChunk(                # CHILD
            segment_id=segment_document.id,       # <-- link child -> parent
            position=position,
            index_node_id=child.metadata.get("doc_id"),
            index_node_hash=child.metadata.get("doc_hash"),
            content=child.page_content,
            word_count=len(child.page_content),
            type="automatic", ...
        ))
```

Nếu segment đã tồn tại (re-index), nó **xoá hết ChildChunk cũ theo `segment_id`**
rồi tạo lại — đảm bảo child luôn đồng bộ với parent.

## 4. Pipeline retrieval (child → parent)

`core/rag/datasource/retrieval_service.py` (hàm tổ chức kết quả, ~dòng 340-462).

1. Vector search trả về các `Document` — thực chất là **child** đã match (vì chỉ
   child được embed). Mỗi kết quả mang `metadata["doc_id"]` = `index_node_id` của
   child và `metadata["score"]`.

2. Với mỗi kết quả, nếu `dataset_document.doc_form == PARENT_CHILD_INDEX`:

   ```python
   child_index_node_id = document.metadata.get("doc_id")
   child_chunk = db.session.scalar(
       select(ChildChunk).where(ChildChunk.index_node_id == child_index_node_id))
   if not child_chunk: continue
   segment = (db.session.query(DocumentSegment)        # tra ra PARENT
       .where(DocumentSegment.id == child_chunk.segment_id,
              DocumentSegment.enabled == True,
              DocumentSegment.status == "completed", ...)
       .first())
   ```

3. **Gộp theo parent** (`segment_child_map`, `include_segment_ids`): nếu nhiều
   child trỏ về cùng một `segment.id`, parent chỉ vào `records` một lần; các child
   còn lại được append vào danh sách `child_chunks` của parent đó, và `score` của
   parent lấy `max` của các child:

   ```python
   if segment.id not in include_segment_ids:
       include_segment_ids.add(segment.id)
       segment_child_map[segment.id] = {"max_score": score,
                                        "child_chunks": [child_chunk_detail]}
       records.append({"segment": segment})
   else:
       segment_child_map[segment.id]["child_chunks"].append(child_chunk_detail)
       segment_child_map[segment.id]["max_score"] = max(..., score)
   ```

4. Đóng gói thành `RetrievalSegments` (`core/rag/embedding/retrieval.py`):

   ```python
   class RetrievalChildChunk(BaseModel):
       id: str; content: str; score: float; position: int

   class RetrievalSegments(BaseModel):
       segment: DocumentSegment                       # PARENT -> đưa cho LLM
       child_chunks: list[RetrievalChildChunk] | None # các child đã match (metadata)
       score: float | None
   ```

`ParentChildIndexProcessor.retrieve()` chỉ gọi `RetrievalService.retrieve(...)` rồi
lọc theo `score_threshold`. **Nội dung đưa vào LLM là `segment.content` (parent)**,
còn child chỉ là thông tin tham chiếu/score.

## 5. Cấu hình & defaults — và một cạm bẫy về "token"

### 5.1 _get_splitter

`core/rag/index_processor/index_processor_base.py`

```python
def _get_splitter(self, processing_rule_mode, max_tokens, chunk_overlap, separator, embedding_model_instance):
    if processing_rule_mode in ["custom", "hierarchical"]:
        max_len = dify_config.INDEXING_MAX_SEGMENTATION_TOKENS_LENGTH   # 4000
        if max_tokens < 50 or max_tokens > max_len:
            raise ValueError(f"Custom segment length should be between 50 and {max_len}.")
        if separator:
            separator = separator.replace("\\n", "\n")
        return FixedRecursiveCharacterTextSplitter.from_encoder(
            chunk_size=max_tokens, chunk_overlap=chunk_overlap,
            fixed_separator=separator, separators=["\n\n", "。", ". ", " ", ""],
            embedding_model_instance=embedding_model_instance)
    else:  # automatic
        return EnhanceRecursiveCharacterTextSplitter.from_encoder(
            chunk_size=DatasetProcessRule.AUTOMATIC_RULES["segmentation"]["max_tokens"],  # 500
            chunk_overlap=...["chunk_overlap"],                                            # 50
            separators=["\n\n", "。", ". ", " ", ""], ...)
```

Parent-child luôn đi nhánh `hierarchical` ⇒ giới hạn `max_tokens` mỗi chunk trong
khoảng **[50, 4000]**.

### 5.2 ⚠️ Cạm bẫy: `max_tokens` thực chất đếm KÝ TỰ, không phải token

`core/rag/splitter/fixed_text_splitter.py`, `from_encoder`:

```python
@classmethod
def from_encoder(cls, embedding_model_instance, ...):
    def _token_encoder(texts):       # đếm theo token thật...
        if embedding_model_instance:
            return embedding_model_instance.get_text_embedding_num_tokens(texts=texts)
        return [GPT2Tokenizer.get_num_tokens(t) for t in texts]
    def _character_encoder(texts):   # ...đếm theo số ký tự
        return [len(t) for t in texts]
    return cls(length_function=_character_encoder, **kwargs)   # <-- DÙNG character_encoder!
```

`_token_encoder` được định nghĩa nhưng **không dùng**; `length_function` thực tế là
`_character_encoder = len(text)`. Vì vậy mọi `max_tokens`/`chunk_overlap` trong
parent-child (và custom mode) **đo bằng số ký tự**, không phải token. Tên field gây
hiểu nhầm. (Đây cũng trùng triết lý "tính theo ký tự `PC_*`" của bản RAG copy trong
`general-agent/`.)

`split_text` của `FixedRecursiveCharacterTextSplitter`: trước tiên `split` theo
`fixed_separator`; mảnh nào dài hơn `chunk_size` mới đệ quy cắt tiếp theo chuỗi
`separators` ưu tiên (`"\n\n"` → `"。"` → `". "` → `" "` → `""`); overlap chỉ áp
dụng ở nhánh ký tự cuối cùng (`separator == ""`).

### 5.3 Defaults & limits

`models/dataset.py` — `DatasetProcessRule.AUTOMATIC_RULES`:

```python
"segmentation": {"delimiter": "\n", "max_tokens": 500, "chunk_overlap": 50}
```

`configs/feature/__init__.py`:

```python
INDEXING_MAX_SEGMENTATION_TOKENS_LENGTH = 4000   # trần chunk (ký tự)
CHILD_CHUNKS_PREVIEW_NUMBER = 50                  # số child tối đa khi preview FULL_DOC
```

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| parent `max_tokens` (automatic) | 500 | thực = 500 ký tự |
| parent `chunk_overlap` (automatic) | 50 | |
| separator mặc định | `"\n"` | thay `\\n`→`\n` trước khi dùng |
| custom/hierarchical `max_tokens` | 50 – 4000 | validate trong `_get_splitter` |
| `parent_mode` mặc định | `"paragraph"` | trong `ParentChildStructureChunk` |
| child preview tối đa (FULL_DOC) | 50 | `CHILD_CHUNKS_PREVIEW_NUMBER` |

> Child không có default cứng riêng — `subchunk_segmentation` do người dùng/khởi
> tạo dataset truyền vào; UI Dify mặc định child nhỏ hơn parent (vd 200 ký tự).

## 6. Vòng đời quản lý child (CRUD thủ công)

Người dùng có thể sửa child sau khi index. Logic ở `services/dataset_service.py`
(`SegmentService`) + `services/vector_service.py` (`VectorService`):

- `create_child_chunk()` → sinh `index_node_id`/hash mới, `type="customized"`, gọi
  `VectorService.create_child_chunk_vector()` để embed.
- `update_child_chunks()` / `update_child_chunk()` → bulk hoặc đơn lẻ, giữ thứ tự
  `position`, regenerate hash + word_count, đẩy lại embedding.
- `delete_child_chunk()` → xoá DB + `delete_child_chunk_vector()` (xoá theo
  `index_node_id` trong vector store).
- `generate_child_chunks()` (vector_service): tái sinh child cho một segment, tạm
  set `parent_mode = FULL_DOC` để coi segment đó như một parent rồi `transform`.

`clean()` của processor: xoá child khỏi vector store (`delete_by_ids` theo
`index_node_id`), tuỳ chọn `delete_child_chunks` để xoá luôn record DB; có
`precomputed_child_node_ids` để tránh race condition khi segment đã bị xoá.

## 7. Tóm tắt luồng end-to-end

**Indexing**

```
File ─extract→ text ─transform→
   [PARAGRAPH]  cắt PARENT (segmentation)  ──► mỗi PARENT cắt CHILD (subchunk_segmentation)
   [FULL_DOC]   cả văn bản = 1 PARENT      ──► cắt CHILD
─load/index→
   PARENT  → DocumentSegment (DB, KHÔNG embed)
   CHILD   → ChildChunk (DB) + Vector store (embed, id = index_node_id)
```

**Retrieval**

```
query → vector search ─match→ CHILD (index_node_id, score)
      → ChildChunk.index_node_id → segment_id → DocumentSegment (PARENT)
      → gộp child cùng parent, score = max(child scores)
      → RetrievalSegments(segment=PARENT, child_chunks=[...])
      → LLM nhận nội dung PARENT
```

## 8. Bảng tra cứu file

| Vai trò | File |
|---|---|
| IndexType enum | `core/rag/index_processor/constant/index_type.py` |
| ParentMode / Rule / Segmentation | `services/entities/knowledge_entities/knowledge_entities.py` |
| Document / ChildDocument / ParentChildStructureChunk | `core/rag/models/document.py` |
| Processor chính | `core/rag/index_processor/processor/parent_child_index_processor.py` |
| Base processor + `_get_splitter` | `core/rag/index_processor/index_processor_base.py` |
| Splitter (character-based) | `core/rag/splitter/fixed_text_splitter.py` |
| Lưu parent/child xuống DB | `core/rag/docstore/dataset_docstore.py` |
| Tra child→parent khi retrieve | `core/rag/datasource/retrieval_service.py` |
| RetrievalSegments / RetrievalChildChunk | `core/rag/embedding/retrieval.py` |
| Models ChildChunk / DocumentSegment / DatasetProcessRule | `models/dataset.py` |
| Migration bảng child_chunks | `migrations/versions/2024_11_22_0701-e19037032219_parent_child_index.py` |
| CRUD child (service) | `services/dataset_service.py`, `services/vector_service.py` |
| Config defaults | `configs/feature/__init__.py`, `models/dataset.py` (`AUTOMATIC_RULES`) |
| Workflow Knowledge Index node | `core/workflow/nodes/knowledge_index/` |
```

## 9. So sánh: splitter cũ (4-stage) ↔ splitter mới (recursive-char kiểu Dify)

> Áp dụng cho `general-agent/`. Kiến trúc parent-child cốt lõi không đổi (chỉ child
> được embed, parent là ngữ cảnh, đo bằng ký tự). Chỉ **thuật toán cắt** thay đổi.

| Khía cạnh | Cũ (4-stage, `blocks.py`+`parent_child.py`) | Mới (recursive-char, `recursive_char.py`) |
|---|---|---|
| Parse cấu trúc | Có — Block heading/table/para | Không — text thuần |
| Ranh giới parent | Cắt theo H1/H2; H3+ gộp vào | Cắt theo `PARENT_SEPARATOR` + size; không biết heading |
| Bảng (table) | Atomic, cắt theo ROW, lặp header | Coi như text thường |
| Chống chunk nhỏ | min-size merge + floor-fold + rescue + consolidate (tới fixpoint) | Không có MIN; có thể sinh nhiều parent nhỏ |
| Cân kích thước | balanced packing (tránh runt) | greedy merge tới sát `chunk_size` |
| Cấu hình | `PC_PARENT_MIN/MAX`, `PC_CHILD_MIN/MAX`, `PC_CHILD_OVERLAP`, `LEVEL3_*` | `PARENT_MODE`, `PARENT_MAX_CHARS`, `CHILD_MAX_CHARS`, `PARENT/CHILD_SEPARATOR`, `RECURSIVE_SEPARATORS` |
| Overlap | có (`PC_CHILD_OVERLAP`, mặc định 40) | không (cố định 0) |
| parent_mode | không (luôn theo heading) | có (`paragraph` / `full-doc`) |
| title | suy ra từ heading | luôn `""` |

**Map config cũ → mới:**

| Cũ | Mới | Ghi chú |
|---|---|---|
| `PC_PARENT_MAX_CHARS=10000` | `PARENT_MAX_CHARS=1024` | đổi default, vẫn là trần parent |
| `PC_PARENT_MIN_CHARS=2000` | (bỏ) | không còn khái niệm min |
| `PC_CHILD_MAX_CHARS=512` | `CHILD_MAX_CHARS=512` | giữ |
| `PC_CHILD_MIN_CHARS=300` | (bỏ) | — |
| `PC_CHILD_OVERLAP=40` | (bỏ) | overlap = 0 |
| `LEVEL3_*` | (bỏ) | không thuộc splitter Dify |
| — | `PARENT_MODE`, `PARENT_SEPARATOR`, `CHILD_SEPARATOR`, `RECURSIVE_SEPARATORS` | mới |

**Ảnh hưởng kỳ vọng:** splitter mới đơn giản, nhanh, không phụ thuộc cấu trúc
markdown; đổi lại mất khả năng giữ bảng nguyên vẹn và có thể tạo nhiều parent nhỏ
(không gộp ở tầng trên). Trên benchmark, recall/precision có thể thay đổi tuỳ
phân bố tài liệu — chạy `eval/local_metrics.py` + LLM judge để đo (xem CLAUDE.md).
