import json
from collections import defaultdict

from common.utils import candidate_span, file_stamp, normalize_box, rerank_document_inputs


SPLITS = ("dev", "test", "test-unseen", "all")
RENDER_DPI = 144


class DocScope:
    candidate_ids = staticmethod(candidate_span)

    def __init__(self, root, *, split="test"):
        self.annotations = root / "benchmark.json"
        self.pdf_dir = root / "pdfs"
        self.split = split
        if split not in SPLITS:
            raise ValueError(f"Unknown DocScope split: {split}")
        if not self.annotations.is_file() or not self.pdf_dir.is_dir():
            raise FileNotFoundError(f"DocScope expects benchmark.json and pdfs/ in {root}. Set --dataset to their directory.")

        import pymupdf

        self.pdf = pymupdf
        with self.annotations.open(encoding="utf-8-sig") as stream:
            entries = json.load(stream)
        source_split = "test" if split == "test-unseen" else split
        excluded_docs = (
            {item["pdf"]["doc_id_str"] for item in entries if item["split"] == "dev"}
            if split == "test-unseen" else set()
        )
        self.entries = [
            item for item in entries
            if (source_split == "all" or item["split"] == source_split)
            and item["answer"]["is_answerable"] and item["pdf"]["doc_id_str"] not in excluded_docs
        ]
        if not self.entries:
            raise ValueError(f"No answerable DocScope queries in split {split}")
        if len({entry["id"] for entry in self.entries}) != len(self.entries):
            raise ValueError("Duplicate DocScope query IDs")
        self.documents = {}

    def pdf_path(self, doc_id):
        return self.pdf_dir / f"{doc_id.removesuffix('.pdf')}.pdf"

    def signature(self):
        doc_ids = sorted({item["pdf"]["doc_id_str"] for item in self.entries})
        return [
            "docscope-1", self.split, RENDER_DPI, file_stamp(self.annotations),
            *[file_stamp(self.pdf_path(doc_id)) for doc_id in doc_ids],
        ]

    def load(self, limit=None, *, include_ocr=False, include_layouts=False):
        docs = defaultdict(list)
        entries = self.entries[:limit]
        for qid, item in enumerate(entries):
            docs[item["pdf"]["doc_id_str"]].append((qid, item))

        queries = [None] * len(entries)
        texts = {} if include_ocr else None
        documents = {}
        start = 0
        for doc_id, items in docs.items():
            path = self.pdf_path(doc_id)
            with self.pdf.open(path) as document:
                end = start + len(document)
                if end == start:
                    raise ValueError(f"Empty DocScope PDF: {path}")
                dimensions = {}
                for qid, item in items:
                    evidence = item["evidences"]
                    pages = {label["page"] for label in evidence}
                    if not pages or any(type(page) is not int or not 1 <= page <= len(document) for page in pages):
                        raise ValueError(f"Invalid DocScope evidence pages for {item['id']}")
                    query = {
                        "qid": qid, "source_qid": item["id"], "query": item["question"],
                        "doc_id": doc_id, "split": item["split"], "group": item["extract_class"],
                        "gt_pages": sorted(start + page - 1 for page in pages),
                        "cand_span": [start, end - 1],
                    }
                    if include_layouts:
                        query["gt_boxes"], query["gt_facts"] = self.evidence_boxes(
                            item, document, start, dimensions,
                        )
                    queries[qid] = query
                if include_ocr:
                    texts.update((uid, page.get_text("text", sort=True)) for uid, page in enumerate(document, start))
                documents[doc_id] = (start, end)
                start = end
        self.documents = documents
        return queries, texts

    def evidence_boxes(self, item, document, start, dimensions):
        page_boxes = defaultdict(set)
        by_id = {}
        scale = RENDER_DPI / 72
        for label in item["evidences"]:
            page_no = label["page"]
            if page_no not in dimensions:
                bounds = (document[page_no - 1].rect * self.pdf.Matrix(scale, scale)).irect
                dimensions[page_no] = (bounds.width, bounds.height)
            box = normalize_box(label["bbox"], dimensions[page_no])
            if box is not None:
                uid = start + page_no - 1
                page_boxes[uid].add(box)
                by_id[label["local_id"]] = (uid, box)

        facts = []
        for fact in item["facts"]:
            evidence_ids = fact.get("evidence_local_ids") or [fact["evidence_local_id"]]
            alternatives = {by_id[label] for label in evidence_ids if label in by_id}
            if alternatives:
                facts.append(sorted(alternatives))
        return {uid: sorted(boxes) for uid, boxes in page_boxes.items()}, facts

    def images(self, ids):
        pending = set(ids)
        for doc_id, (start, end) in self.documents.items():
            selected = sorted(pending.intersection(range(start, end)))
            if selected:
                with self.pdf.open(self.pdf_path(doc_id)) as document:
                    for uid in selected:
                        image = document[uid - start].get_pixmap(dpi=RENDER_DPI, alpha=False, annots=False)
                        yield uid, image.tobytes("jpeg", jpg_quality=90)
                pending.difference_update(selected)
            if not pending:
                break
        if pending:
            raise IndexError("Image page UID outside the loaded DocScope documents")

    def rerank_inputs(self, queries, rankings):
        return rerank_document_inputs(self.images, queries, rankings)
