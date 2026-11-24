import json
from collections import defaultdict

import pymupdf

from utils import candidate_span, normalize_box, report_progress, rerank_document_inputs


SPLITS = ("dev", "test", "test-unseen", "all")
RENDER_DPI = 144


class DocScope:
    candidate_ids = staticmethod(candidate_span)

    def __init__(self, root, *, split="test"):
        self.pdf_dir = root / "pdfs"
        with (root / "benchmark.json").open(encoding="utf-8-sig") as stream:
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
        self.documents = {}

    def open_pdf(self, doc_id):
        document = pymupdf.open(self.pdf_dir / f"{doc_id.removesuffix('.pdf')}.pdf")
        catalog = document.pdf_catalog()
        if (
            document.xref_get_key(catalog, "OCProperties/OCGs")[0] == "array"
            and document.xref_get_key(catalog, "OCProperties/D")[0] == "null"
        ):
            # Match MuPDF's fallback for missing layer configuration.
            document.xref_set_key(catalog, "OCProperties/D", "<< /BaseState /ON >>")
        return document

    def load(self, limit=None, *, include_ocr=False, include_layouts=False):
        docs = defaultdict(list)
        entries = self.entries[:limit]
        for qid, item in enumerate(entries):
            docs[item["pdf"]["doc_id_str"]].append((qid, item))

        queries = [None] * len(entries)
        texts = {} if include_ocr else None
        documents = {}
        start = 0
        label = "Reading DocScope PDF text" if include_ocr else "Checking DocScope PDFs"
        count = len(docs)
        report_progress(label, 0, count, every=10)
        for completed, (doc_id, items) in enumerate(docs.items(), 1):
            with self.open_pdf(doc_id) as document:
                end = start + len(document)
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
            report_progress(label, completed, count, every=10)
        self.documents = documents
        return queries, texts

    @staticmethod
    def evidence_boxes(item, document, start, dimensions):
        page_boxes = defaultdict(set)
        by_id = {}
        scale = RENDER_DPI / 72
        for label in item["evidences"]:
            page_no = label["page"]
            if page_no not in dimensions:
                bounds = (document[page_no - 1].rect * pymupdf.Matrix(scale, scale)).irect
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
                with self.open_pdf(doc_id) as document:
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
