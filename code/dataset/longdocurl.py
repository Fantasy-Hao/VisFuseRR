import re
import tarfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import PurePosixPath
from shutil import copyfileobj

import pymupdf

from utils import atomic_output, candidate_span, normalize_box, read_jsonl, rerank_document_inputs


ANNOTATIONS = ("LongDocURL_public_with_subtask_category.jsonl", "LongDocURL_public.jsonl")
RENDER_DPI = 144
BOX = re.compile(r"<box>\s*([^<>]*?)\s*</box>", re.IGNORECASE)
PAGE_BEFORE_BOX = re.compile(r"\bpage\s+(\d+)\s*[:(]?\s*$", re.IGNORECASE)
PAGE_AFTER_BOX = re.compile(r"\s*(?:,\s*|on\s+)page\s+(\d+)\b", re.IGNORECASE)


class LongDocURL:
    candidate_ids = staticmethod(candidate_span)

    def __init__(self, root):
        self.pdf_dir = root / "pdfs"
        self.archive = root / "pdf_files.tar.gz"
        self.annotations = next(
            (root / name for name in ANNOTATIONS if (root / name).is_file()),
            root / ANNOTATIONS[0],
        )
        self.documents = {}

    def load(self, limit=None, *, include_ocr=False, include_layouts=False):
        entries = [item for item in read_jsonl(self.annotations) if item["evidence_pages"]][:limit]
        if not entries:
            raise ValueError("No LongDocURL queries with annotated evidence pages")

        documents = {}
        queries = []
        next_uid = 0
        for item in entries:
            doc_id = item["doc_no"]
            total = item["total_pages"]
            if re.fullmatch(r"[0-9]{7}", doc_id) is None or type(total) is not int or total < 1:
                raise ValueError(f"Invalid LongDocURL document metadata: {doc_id}")
            start, end = documents.get(doc_id, (next_uid, next_uid + total))
            if doc_id not in documents:
                documents[doc_id] = start, end
                next_uid = end
            if end - start != total:
                raise ValueError(f"Inconsistent LongDocURL page count: {doc_id}")

            pages = item["evidence_pages"]
            if any(type(page) is not int or not 1 <= page <= total for page in pages):
                raise ValueError(f"Invalid LongDocURL evidence pages for {item['question_id']}")
            query = {
                "qid": len(queries), "source_qid": item["question_id"], "doc_id": doc_id,
                "query": item["question"], "group": item["task_tag"], "cand_span": [start, end - 1],
                "gt_pages": sorted({start + page - 1 for page in pages}),
                "content_type": item["evidence_sources"],
                "subtasks": item.get("subTask", []),
            }
            if include_layouts:
                query["gt_boxes"] = self.evidence_boxes(item["detailed_evidences"], set(pages), start)
            queries.append(query)

        texts = None
        if include_ocr:
            texts = {}
            self.prepare_pdfs(documents)
            for doc_id, (start, end) in documents.items():
                with self.open_pdf(doc_id, end - start) as document:
                    texts.update((uid, page.get_text("text", sort=True)) for uid, page in enumerate(document, start))
        self.documents = documents
        return queries, texts

    @staticmethod
    def evidence_boxes(text, pages, start):
        page_boxes = defaultdict(set)
        for match in BOX.finditer(text):
            before = PAGE_BEFORE_BOX.search(text, max(0, match.start() - 40), match.start())
            after = PAGE_AFTER_BOX.match(text, match.end())
            if after and re.match(r"\s*(?:[-–:]|(?:,|and\b|to\b)\s*\d)", text[after.end():]):
                after = None
            cited = {int(citation[1]) for citation in (before, after) if citation}
            # A free-text citation needs one unambiguous adjacent page.
            if len(cited) != 1 or not cited <= pages:
                continue
            coords = match[1]
            if coords.startswith("(") and coords.endswith(")") or coords.startswith("[") and coords.endswith("]"):
                coords = coords[1:-1]
            try:
                box = normalize_box(tuple(float(value) for value in coords.split(",")), (1, 1))
            except ValueError:
                continue
            if box is not None:
                uid = start + cited.pop() - 1
                page_boxes[uid].add(box)
        return {uid: sorted(boxes) for uid, boxes in page_boxes.items()}

    def prepare_pdfs(self, doc_ids):
        pending = {f"{doc_id}.pdf" for doc_id in doc_ids if not (self.pdf_dir / f"{doc_id}.pdf").is_file()}
        if pending and self.archive.is_file():
            with tarfile.open(self.archive, "r|gz") as archive:
                for member in archive:
                    name = PurePosixPath(member.name).name
                    if not member.isfile() or name not in pending:
                        continue
                    # Extract only the expected basename, never an archive path.
                    with archive.extractfile(member) as source, atomic_output(self.pdf_dir / name, "wb") as output:
                        copyfileobj(source, output)
                    pending.remove(name)
                    if not pending:
                        break
        if pending:
            raise FileNotFoundError(f"Missing LongDocURL PDF {min(pending)} in {self.pdf_dir} or {self.archive}")

    @contextmanager
    def open_pdf(self, doc_id, expected_pages):
        with pymupdf.open(self.pdf_dir / f"{doc_id}.pdf") as document:
            if len(document) != expected_pages:
                raise ValueError(f"LongDocURL PDF page count does not match annotations: {doc_id}")
            yield document

    def images(self, ids):
        pending = set(ids)
        if not pending:
            return
        selected = {}
        for doc_id, (start, end) in self.documents.items():
            pages = pending.intersection(range(start, end))
            if pages:
                selected[doc_id] = (start, end, sorted(pages))
                pending.difference_update(pages)
        if pending:
            raise IndexError("Image page UID outside the loaded LongDocURL documents")
        self.prepare_pdfs(selected)
        for doc_id, (start, end, pages) in selected.items():
            with self.open_pdf(doc_id, end - start) as document:
                for uid in pages:
                    image = document[uid - start].get_pixmap(dpi=RENDER_DPI, alpha=False, annots=False)
                    yield uid, image.tobytes("jpeg", jpg_quality=90)

    def rerank_inputs(self, queries, rankings):
        return rerank_document_inputs(self.images, queries, rankings)
