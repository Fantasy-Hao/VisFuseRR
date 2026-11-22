from itertools import groupby, islice
from math import isfinite

import pyarrow.parquet as pq

from common.utils import ROOT, file_stamp, read_jsonl


def normalize_box(box, page_size):
    if (
        not isinstance(box, (list, tuple)) or len(box) != 4
        or not isinstance(page_size, (list, tuple)) or len(page_size) != 2
        or not all(type(value) in (int, float) and isfinite(value) for value in (*box, *page_size))
    ):
        return None
    x1, y1, x2, y2 = box
    width, height = page_size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    normalized = tuple(round(value / side * 1000) for value, side in zip(box, (width, height, width, height)))
    if normalized[0] < normalized[2] and normalized[1] < normalized[3]:
        return normalized
    return None


class MMDocIR:
    def __init__(self, root):
        self.annotations = root / "MMDocIR_annotations.jsonl"
        self.pages_path = root / "MMDocIR_pages.parquet"
        self.layouts_path = root / "MMDocIR_layouts.parquet"
        self.pages = pq.ParquetFile(self.pages_path)

    def signature(self):
        return [file_stamp(self.annotations), file_stamp(self.pages_path)]

    def load(self, limit=None, *, include_ocr=False, include_layouts=False):
        columns = ["doc_name", "passage_id"]
        if include_ocr:
            columns.append("ocr_text")
        pages = self.pages.read(columns=columns).to_pylist()
        texts = [page["ocr_text"] or "" for page in pages] if include_ocr else None
        layouts = pq.read_table(
            self.layouts_path, columns=["doc_name", "page_id", "bbox", "page_size"],
        ) if include_layouts else None
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
                layout_boxes = self.document_layout_boxes(doc, layouts)
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
                if layouts is not None:
                    queries[-1]["gt_boxes"] = self.evidence_boxes(item, local_to_global, set(gt), layout_boxes)
                if len(queries) == limit:
                    return queries, texts
        if not queries:
            raise ValueError("No MMDocIR queries found")
        return queries, texts

    @staticmethod
    def document_layout_boxes(doc, layouts):
        start, end = doc["layout_indices"]
        name = doc["doc_name"]
        if start < 0 or end < start or end >= len(layouts):
            raise ValueError(f"Invalid document layout span: {name}")
        boxes = {}
        for row in layouts.slice(start, end - start + 1).to_pylist():
            if row["doc_name"] not in (name, name.removesuffix(".pdf")):
                raise ValueError(f"Invalid document layout span: {name}")
            box = normalize_box(row["bbox"], row["page_size"])
            if box is not None:
                key = row["page_id"], tuple(row["bbox"]), tuple(row["page_size"])
                boxes[key] = box
        return boxes

    @staticmethod
    def evidence_boxes(question, local_to_global, relevant_pages, layout_boxes):
        boxes = {}
        for label in question.get("layout_mapping") or []:
            page = int(label["page"])
            uid = local_to_global.get(page)
            if uid not in relevant_pages:
                continue
            annotated = normalize_box(label["bbox"], label["page_size"])
            if annotated is None:
                continue
            key = page, tuple(label["bbox"]), tuple(label["page_size"])
            # Query annotations may cover several layout rows or have no exact match.
            box = layout_boxes.get(key, annotated)
            boxes.setdefault(uid, set()).add(box)
        return {uid: sorted(page_boxes) for uid, page_boxes in boxes.items()}

    def candidate_ids(self, query):
        start, end = query["cand_span"]
        return range(start, end + 1)

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
        """Stream each document once and yield a query with its selected image bytes."""
        source = self.images({uid for ids in rankings.values() for uid in ids})
        by_span = sorted(queries, key=lambda query: query["cand_span"])
        for _, group in groupby(by_span, key=lambda query: query["cand_span"]):
            document_queries = list(group)
            ids = {uid for query in document_queries for uid in rankings[query["qid"]]}
            images = dict(islice(source, len(ids)))
            if images.keys() != ids:
                raise ValueError("Candidate images do not match document pages")
            for query in document_queries:
                yield query, [images[uid] for uid in rankings[query["qid"]]]


class ViDoReV3:
    """Reserved adapter for the corpus, queries and qrels splits."""

    def __init__(self, root):
        self.root = root

    def load(self, limit=None, *, include_ocr=False, include_layouts=False):
        raise NotImplementedError("ViDoRe V3 is reserved. Select --dataset-name mmdocir.")


DATASETS = {"mmdocir": MMDocIR, "vidore_v3": ViDoReV3}


def open_dataset(name, directory=None):
    return DATASETS[name](directory or ROOT / "data" / name)
