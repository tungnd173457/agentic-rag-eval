"""Nạp gold-docs (.txt) vào Weaviate LOCAL — tái dùng chunking + embedding THẬT.

Đây là bản "lean ingest": bỏ qua S3 / Mongo / phân loại domain (LLM) của pipeline
gốc, nhưng GIỮ NGUYÊN bộ phận đo retrieval:
  - chunking: ingestion.splitters.split_document (parent/child theo heading)
  - embedding: services.embedding.embed_texts (đúng provider/model trong .env)
  - vectordb: services.weaviate.upsert_chunks (đúng Chunk schema mà agent đọc)

doc_id = phần uuid 32-hex của tên file (bỏ tiền tố 'dsid_'), để chunk_id =
doc_id[:8]#pN khớp regex citation HEX của orchestrator. Khi chấm điểm ta map
ngược doc_id -> 'dsid_' + doc_id để so với expected_doc_ids của benchmark.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import bootstrap  # noqa: F401  (phải import trước config)
import eval_config as C
from ingestion.splitters import split_document
from services.embedding import embed_texts
from services import weaviate as wrepo

_GENERAL = "general_text"


def doc_id_from_filename(filename: str) -> str:
    """'dsid_<uuid>__<slug>.txt' -> '<uuid>' (32-hex)."""
    stem = filename[:-4] if filename.endswith(".txt") else filename
    token = stem.split("__", 1)[0]
    return token[len("dsid_"):] if token.startswith("dsid_") else token


def _chunk_objects(doc_id: str, filename: str, parents, children, vectors) -> list[dict]:
    updated_at = datetime.now(UTC).isoformat()
    title_by_parent = {p.parent_id: p.title for p in parents}
    common = {
        "doc_id": doc_id, "s3_key": "", "filename": filename,
        "domain_level_1": _GENERAL, "domain_level_2": _GENERAL,
        "domain_level_3": _GENERAL, "domains_level_3": [_GENERAL],
        "domain_tags_level_3": [], "updated_at": updated_at,
    }
    objs: list[dict] = []
    for p in parents:
        objs.append({
            "id": wrepo.chunk_uuid(doc_id, p.parent_id),
            "properties": {
                **common, "kind": "parent", "chunk_id": p.parent_id, "parent_id": "",
                "position": p.position, "char_count": p.char_count,
                "doc_hash": p.doc_hash, "title": p.title, "text": p.text,
            },
        })
    for c, vec in zip(children, vectors):
        objs.append({
            "id": wrepo.chunk_uuid(doc_id, c.child_id),
            "vector": vec,
            "properties": {
                **common, "kind": "child", "chunk_id": c.child_id,
                "parent_id": c.parent_id, "position": c.position,
                "char_count": c.char_count, "doc_hash": c.doc_hash,
                "title": title_by_parent.get(c.parent_id, ""), "text": c.text,
            },
        })
    return objs


def ingest_one(path: Path) -> tuple[str, int, int]:
    doc_id = doc_id_from_filename(path.name)
    markdown = path.read_text(encoding="utf-8", errors="replace")
    parents, children = split_document(markdown)
    if not parents:
        return doc_id, 0, 0
    vectors = embed_texts([c.text for c in children]) if children else []
    wrepo.delete_chunks(doc_id)  # idempotent: re-run sạch chunk cũ
    objs = _chunk_objects(doc_id, path.name, parents, children, vectors)
    wrepo.upsert_chunks(objs)
    return doc_id, len(parents), len(children)


def main() -> None:
    ap = argparse.ArgumentParser(description="Lean-ingest gold-docs vào Weaviate local")
    ap.add_argument("--docs-dir", default=str(C.GOLD_DOCS_DIR))
    ap.add_argument("--limit", type=int, default=None, help="Chỉ nạp N file đầu (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="Bỏ qua doc đã có trong manifest (theo doc_id)")
    args = ap.parse_args()

    print("Cấu hình:", json.dumps(bootstrap.info(), ensure_ascii=False, indent=2))
    wrepo.ensure_chunk_schema()

    files = sorted(Path(args.docs_dir).glob("*.txt"))
    if args.limit:
        files = files[: args.limit]

    manifest: dict[str, dict] = {}
    if args.resume and C.INGEST_MANIFEST_FILE.exists():
        manifest = json.loads(C.INGEST_MANIFEST_FILE.read_text())

    print(f"Nạp {len(files)} file từ {args.docs_dir}")
    t0 = time.perf_counter()
    total_p = total_c = done = skipped = failed = 0
    for i, path in enumerate(files, 1):
        did = doc_id_from_filename(path.name)
        if args.resume and did in manifest:
            skipped += 1
            continue
        try:
            doc_id, np_, nc = ingest_one(path)
            manifest[doc_id] = {"filename": path.name, "parents": np_, "children": nc}
            total_p += np_
            total_c += nc
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ! LỖI {path.name}: {exc}")
        if i % 25 == 0 or i == len(files):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(files)}] ok={done} skip={skipped} fail={failed} "
                  f"parents={total_p} children={total_c} ({el:.0f}s)")
        # ghi manifest định kỳ để --resume an toàn nếu ngắt giữa chừng
        if i % 50 == 0:
            C.INGEST_MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    C.INGEST_MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"XONG: {done} doc, {total_p} parents, {total_c} children, fail={failed}. "
          f"Manifest → {C.INGEST_MANIFEST_FILE}")


if __name__ == "__main__":
    main()
