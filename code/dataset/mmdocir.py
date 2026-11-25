"""MMDocIR dataset adapter."""

import pyarrow.parquet as pq

from utils import candidate_span, normalize_box, read_jsonl


class MMDocIR:
    candidate_ids = staticmethod(candidate_span)

    def __init__(self, root):
        self.annotations = root / "MMDocIR_annotations.jsonl"
        self.pages_path = root / "MMDocIR_pages.parquet"
        self.layouts_path = root / "MMDocIR_layouts.parquet"

    def load(self, limit=None, *, include_ocr=False, include_layouts=False):
        columns = ["doc_name", "passage_id"]
        if include_ocr:
            columns.append("ocr_text")
        pages = pq.read_table(self.pages_path, columns=columns).to_pylist()
        texts = [page["ocr_text"] or "" for page in pages] if include_ocr else None
        layouts = None
        if include_layouts:
            layouts = pq.read_table(self.layouts_path, columns=["doc_name", "page_id", "bbox", "page_size"])
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
            if layouts is not None:
                layout_boxes = self.document_layouts(doc, layouts)
            for item in doc["questions"]:
                gt = sorted({local_to_global[int(page)] for page in item["page_id"]})
                if not gt:
                    raise ValueError(f"Query has no ground-truth page: {name}")
                content_type = item.get("type")
                if isinstance(content_type, list):
                    content_type = next(iter(content_type), None)
                query = {
                    "qid": len(queries), "query": item["Q"], "group": doc["domain"],
                    "gt_pages": gt, "cand_span": [start, end],
                    "content_type": content_type,
                }
                if layouts is not None:
                    query["gt_boxes"] = self.evidence_boxes(item, local_to_global, set(gt), layout_boxes)
                queries.append(query)
                if len(queries) == limit:
                    return queries, texts
        return queries, texts

    @staticmethod
    def document_layouts(doc, layouts):
        start, end = doc["layout_indices"]
        name = doc["doc_name"]
        if not 0 <= start <= end < len(layouts):
            raise ValueError(f"Invalid document layout span: {name}")
        boxes = {}
        for row in layouts.slice(start, end - start + 1).to_pylist():
            if row["doc_name"] not in (name, name.removesuffix(".pdf")):
                raise ValueError(f"Invalid document layout span: {name}")
            if row["bbox"] and row["page_size"]:
                boxes[row["page_id"], tuple(row["bbox"])] = row["page_size"]
        return boxes

    @staticmethod
    def evidence_boxes(question, local_to_global, relevant_pages, layout_boxes):
        boxes = {}
        for label in question.get("layout_mapping") or []:
            page = int(label["page"])
            uid = local_to_global.get(page)
            if uid not in relevant_pages:
                continue
            coordinates = label.get("bbox")
            size = label.get("page_size") or layout_boxes.get((page, tuple(coordinates or [])))
            box = normalize_box(coordinates, size)
            if box is not None:
                boxes.setdefault(uid, set()).add(box)
        return {uid: sorted(page_boxes) for uid, page_boxes in boxes.items()}

    def images(self, ids):
        """Yield selected (page ID, image bytes) in page order."""
        pending = set(ids)
        if not pending:
            return
        with pq.ParquetFile(self.pages_path) as pages:
            if min(pending) < 0 or max(pending) >= pages.metadata.num_rows:
                raise IndexError("Image page UID outside parquet")
            offset = 0
            for batch in pages.iter_batches(batch_size=64, columns=["image_binary"]):
                end = offset + len(batch)
                for uid in sorted(pending.intersection(range(offset, end))):
                    binary = batch.column(0)[uid - offset].as_py()
                    if not binary:
                        raise ValueError(f"Missing image data for MMDocIR page {uid}")
                    yield uid, binary
                    pending.remove(uid)
                if not pending:
                    return
                offset = end
