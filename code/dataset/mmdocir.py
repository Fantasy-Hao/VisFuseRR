import pyarrow.parquet as pq

from utils import candidate_span, normalize_box, read_jsonl, rerank_document_inputs


class MMDocIR:
    candidate_ids = staticmethod(candidate_span)

    def __init__(self, root):
        self.annotations = root / "MMDocIR_annotations.jsonl"
        self.pages = pq.ParquetFile(root / "MMDocIR_pages.parquet")

    def load(self, limit=None, *, include_ocr=False, include_layouts=False):
        columns = ["doc_name", "passage_id"]
        if include_ocr:
            columns.append("ocr_text")
        pages = self.pages.read(columns=columns).to_pylist()
        texts = [page["ocr_text"] or "" for page in pages] if include_ocr else None
        queries = []
        for doc in read_jsonl(self.annotations):
            start, end = doc["page_indices"]
            rows = pages[start:end + 1]
            name = doc["doc_name"]
            if start < 0 or end < start or len(rows) != end - start + 1 or any(
                row["doc_name"] not in (name, name.removesuffix(".pdf")) for row in rows
            ):
                raise ValueError(f"Invalid document row span: {name}")
            local_to_global = {int(row["passage_id"]): uid for uid, row in enumerate(rows, start)}
            if len(local_to_global) != len(rows):
                raise ValueError(f"Duplicate local page numbers: {name}")
            for item in doc["questions"]:
                gt = sorted({local_to_global[int(page)] for page in item["page_id"]})
                if not gt:
                    raise ValueError(f"Query has no ground-truth page: {name}")
                content_type = item.get("type")
                if isinstance(content_type, list):
                    content_type = next(iter(content_type), None)
                queries.append({
                    "qid": len(queries), "query": item["Q"], "group": doc["domain"],
                    "gt_pages": gt, "cand_span": [start, end],
                    "content_type": content_type,
                })
                if include_layouts:
                    queries[-1]["gt_boxes"] = self.evidence_boxes(item, local_to_global, set(gt))
                if len(queries) == limit:
                    return queries, texts
        return queries, texts

    @staticmethod
    def evidence_boxes(question, local_to_global, relevant_pages):
        boxes = {}
        for label in question.get("layout_mapping") or []:
            page = int(label["page"])
            uid = local_to_global.get(page)
            if uid not in relevant_pages:
                continue
            box = normalize_box(label["bbox"], label["page_size"])
            if box is not None:
                boxes.setdefault(uid, set()).add(box)
        return {uid: sorted(page_boxes) for uid, page_boxes in boxes.items()}

    def images(self, ids):
        """Yield selected (page ID, image bytes) in page order."""
        pending = set(ids)
        if not pending:
            return
        if min(pending) < 0 or max(pending) >= self.pages.metadata.num_rows:
            raise IndexError("Image page UID outside parquet")
        offset = 0
        for batch in self.pages.iter_batches(batch_size=64, columns=["image_binary"]):
            for uid in sorted(pending.intersection(range(offset, offset + len(batch)))):
                yield uid, batch.column(0)[uid - offset].as_py()
                pending.remove(uid)
            offset += len(batch)
            if not pending:
                break

    def rerank_inputs(self, queries, rankings):
        return rerank_document_inputs(self.images, queries, rankings)
