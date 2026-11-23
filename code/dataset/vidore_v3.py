from collections import defaultdict
from io import BytesIO
from tempfile import TemporaryFile

import pyarrow.parquet as pq
from PIL import Image

from utils import candidate_span, group_candidates, normalize_box


SUBSETS = ("all", "computer_science", "finance_en", "hr", "industrial", "pharmaceuticals")


class ViDoReV3:
    candidate_ids = staticmethod(candidate_span)

    def __init__(self, root, *, subset="all"):
        if subset not in SUBSETS:
            raise ValueError(f"Unknown ViDoRe V3 subset: {subset}")
        self.root = root
        self.subsets = SUBSETS[1:] if subset == "all" else (subset,)
        self.image_files = []

    def load(self, limit=None, *, include_ocr=False, include_layouts=False):
        queries = []
        texts = [] if include_ocr else None
        self.image_files = []
        next_page = 0
        for group in self.subsets:
            directory = self.root / f"vidore_v3_{group}"
            if self.root.name == directory.name:
                directory = self.root
            first_page = next_page
            page_ids = {}
            for shard in sorted((directory / "corpus").glob("test-*.parquet")):
                columns = ["corpus_id", "markdown"] if include_ocr else ["corpus_id"]
                pages = pq.read_table(shard, columns=columns)
                page_ids.update((pid, uid) for uid, pid in enumerate(pages["corpus_id"].to_pylist(), next_page))
                if include_ocr:
                    texts.extend(text or "" for text in pages["markdown"].to_pylist())
                self.image_files.append((next_page, next_page + len(pages), shard))
                next_page += len(pages)

            items = pq.read_table(
                directory / "queries", columns=["query_id", "query", "content_type"],
                filters=[("language", "=", "english")],
            )
            if limit is not None:
                items = items.slice(0, limit - len(queries))
            by_source = {}
            for item in items.to_pylist():
                query = {
                    "qid": len(queries), "source_qid": item["query_id"], "query": item["query"],
                    "group": group, "content_type": item["content_type"],
                    "cand_span": [first_page, next_page - 1], "gt_relevance": {},
                }
                if include_layouts:
                    query["gt_boxes"] = {}
                by_source[item["query_id"]] = query
                queries.append(query)

            columns = ["query_id", "corpus_id", "score"]
            if include_layouts:
                columns.append("bounding_boxes")
            qrels = pq.read_table(
                directory / "qrels", columns=columns,
                filters=[("query_id", "in", list(by_source)), ("score", ">", 0)],
            )
            labels = defaultdict(list)
            for row in qrels.to_pylist():
                query = by_source[row["query_id"]]
                uid = page_ids[row["corpus_id"]]
                query["gt_relevance"][uid] = row["score"]
                if include_layouts and row["bounding_boxes"]:
                    labels[uid].append((query, row["bounding_boxes"]))
            for query in by_source.values():
                query["gt_pages"] = sorted(query["gt_relevance"])
            self.add_boxes(labels)
            if len(queries) == limit:
                break
        return queries, texts

    def add_boxes(self, labels):
        for uid, binary in self.images(labels):
            with Image.open(BytesIO(binary)) as image:
                size = image.size
            for query, regions in labels[uid]:
                boxes = set()
                for region in regions:
                    coords = (region["x1"], region["y1"], region["x2"], region["y2"])
                    box = normalize_box(coords, size)
                    if box is not None:
                        boxes.add(box)
                if boxes:
                    query["gt_boxes"][uid] = sorted(boxes)

    def images(self, ids):
        pending = set(ids)
        for start, end, shard in self.image_files:
            selected = pending.intersection(range(start, end))
            if not selected:
                continue
            with pq.ParquetFile(shard) as pages:
                offset = start
                for batch in pages.iter_batches(batch_size=32, columns=["image"]):
                    batch_end = offset + len(batch)
                    for uid in sorted(selected.intersection(range(offset, batch_end))):
                        yield uid, batch.column(0)[uid - offset]["bytes"].as_py()
                        selected.remove(uid)
                    if not selected:
                        break
                    offset = batch_end

    def rerank_inputs(self, queries, rankings):
        for group in group_candidates(queries):
            ids = {uid for query in group for uid in rankings[query["qid"]]}
            # Store each candidate once; a full subset can contain several GB of page images.
            with TemporaryFile() as cache:
                positions = {}
                for uid, binary in self.images(ids):
                    positions[uid] = cache.tell(), len(binary)
                    cache.write(binary)
                for query in group:
                    images = []
                    for uid in rankings[query["qid"]]:
                        offset, length = positions[uid]
                        cache.seek(offset)
                        images.append(cache.read(length))
                    yield query, images
