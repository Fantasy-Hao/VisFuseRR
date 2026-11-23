from collections import defaultdict
from io import BytesIO
from itertools import islice
from tempfile import TemporaryFile

import pyarrow.parquet as pq
from PIL import Image

from common.utils import candidate_span, file_stamp, group_candidates, normalize_box


NATIVE_LANGUAGES = {
    "computer_science": "english",
    "energy": "french",
    "finance_en": "english",
    "finance_fr": "french",
    "hr": "english",
    "industrial": "english",
    "pharmaceuticals": "english",
    "physics": "french",
}
SUBSETS = ("all", *NATIVE_LANGUAGES)
LANGUAGES = ("native", "all", "english", "french", "german", "italian", "portuguese", "spanish")
TABLES = ("corpus", "queries", "qrels")


class ViDoReV3:
    candidate_ids = staticmethod(candidate_span)

    def __init__(self, root, *, subset="all", language="native"):
        if subset not in SUBSETS or language not in LANGUAGES:
            raise ValueError(f"Unknown ViDoRe V3 selection: {subset}/{language}")
        self.language = language
        self.corpora = {}
        for name in NATIVE_LANGUAGES if subset == "all" else (subset,):
            folder = f"vidore_v3_{name}"
            direct = root.name == folder or (subset == name and (root / "corpus").is_dir())
            path = root if direct else root / folder
            if path.is_dir():
                shards = {table: sorted((path / table).glob("test-*.parquet")) for table in TABLES}
                if not all(shards.values()):
                    raise FileNotFoundError(f"ViDoRe V3 expects corpus, queries and qrels test parquet shards in {path}")
                self.corpora[name] = shards
        if not self.corpora:
            raise FileNotFoundError(f"No ViDoRe V3 corpus for {subset} in {root}")
        self.image_files = []

    def signature(self):
        return [
            "vidore-v3-1", self.language,
            *[file_stamp(path) for shards in self.corpora.values() for paths in shards.values() for path in paths],
        ]

    def load(self, limit=None, *, include_ocr=False, include_layouts=False):
        queries = []
        texts = {} if include_ocr else None
        self.image_files = []
        next_page = next_query = 0
        for group, shards in self.corpora.items():
            first_page = next_page
            ids = {}
            for path in shards["corpus"]:
                columns = ["corpus_id", "markdown"] if include_ocr else ["corpus_id"]
                rows = pq.read_table(path, columns=columns).to_pylist()
                for uid, row in enumerate(rows, next_page):
                    corpus_id = row["corpus_id"]
                    if corpus_id in ids:
                        raise ValueError(f"Duplicate ViDoRe V3 page ID {corpus_id} in {group}")
                    ids[corpus_id] = uid
                    if include_ocr:
                        texts[uid] = row["markdown"] or ""
                self.image_files.append((next_page, next_page + len(rows), path))
                next_page += len(rows)

            items = pq.read_table(shards["queries"]).to_pylist()
            seen = {item["query_id"] for item in items}
            if len(seen) != len(items):
                raise ValueError(f"Duplicate ViDoRe V3 query IDs in {group}")
            language = NATIVE_LANGUAGES[group] if self.language == "native" else self.language
            selected = [
                (qid, item) for qid, item in enumerate(items, next_query)
                if language == "all" or item["language"] == language
            ]
            next_query += len(items)
            if limit is not None:
                selected = selected[:limit - len(queries)]

            qrels = {item["query_id"]: {} for _, item in selected}
            labels = defaultdict(list)
            columns = ["query_id", "corpus_id", "score"]
            if include_layouts:
                columns.append("bounding_boxes")
            for row in pq.read_table(shards["qrels"], columns=columns).to_pylist():
                relevant = qrels.get(row["query_id"])
                if relevant is None or row["score"] == 0:
                    continue
                uid = ids.get(row["corpus_id"])
                score = row["score"]
                if uid is None or score not in (1, 2) or uid in relevant:
                    raise ValueError(f"Invalid ViDoRe V3 qrel in {group}: {row['query_id']}/{row['corpus_id']}")
                relevant[uid] = score
                if include_layouts and row["bounding_boxes"]:
                    labels[uid].append((row["query_id"], row["bounding_boxes"]))

            group_queries = {}
            for qid, item in selected:
                source_qid = item["query_id"]
                relevant = qrels[source_qid]
                if not relevant:
                    raise ValueError(f"No positive ViDoRe V3 qrels for {group}/{source_qid}")
                query = {
                    "qid": qid, "source_qid": source_qid, "query": item["query"], "group": group,
                    "language": item["language"], "content_type": item["content_type"],
                    "gt_pages": sorted(relevant), "gt_relevance": relevant,
                    "cand_span": [first_page, next_page - 1],
                }
                if include_layouts:
                    query["gt_boxes"] = {}
                    group_queries[source_qid] = query
                queries.append(query)
            if labels:
                self.add_boxes(group_queries, labels)
            if len(queries) == limit:
                break
        if not queries:
            raise ValueError(f"No ViDoRe V3 queries for language {self.language}")
        return queries, texts

    def add_boxes(self, queries, labels):
        for uid, binary in self.images(labels):
            with Image.open(BytesIO(binary)) as page:
                size = page.size
            for source_qid, rows in labels[uid]:
                boxes = set()
                for row in rows:
                    box = normalize_box((row["x1"], row["y1"], row["x2"], row["y2"]), size)
                    if box is not None:
                        boxes.add(box)
                if boxes:
                    queries[source_qid]["gt_boxes"][uid] = sorted(boxes)

    def images(self, ids):
        pending = sorted(set(ids))
        index = 0
        for start, end, path in self.image_files:
            if index == len(pending):
                break
            if pending[index] >= end:
                continue
            if pending[index] < start:
                raise IndexError("Image page UID outside the loaded ViDoRe V3 corpora")
            with pq.ParquetFile(path) as pages:
                offset = start
                for row_group in range(pages.num_row_groups):
                    if index == len(pending) or pending[index] >= end:
                        break
                    group_end = offset + pages.metadata.row_group(row_group).num_rows
                    if pending[index] < group_end:
                        for batch in pages.iter_batches(batch_size=32, row_groups=[row_group], columns=["image"]):
                            batch_end = offset + len(batch)
                            while index < len(pending) and pending[index] < batch_end:
                                uid = pending[index]
                                binary = batch.column(0)[uid - offset]["bytes"].as_py()
                                if not binary:
                                    raise ValueError(f"Missing ViDoRe V3 image bytes for page {uid}")
                                yield uid, binary
                                index += 1
                            offset = batch_end
                            if index == len(pending) or pending[index] >= group_end:
                                break
                    offset = group_end
        if index != len(pending):
            raise IndexError("Image page UID outside the loaded ViDoRe V3 corpora")

    def rerank_inputs(self, queries, rankings):
        source = self.images({uid for ids in rankings.values() for uid in ids})
        for group in group_candidates(queries):
            ids = {uid for query in group for uid in rankings[query["qid"]]}
            # A corpus can have several GB of candidate images shared across queries.
            with TemporaryFile() as image_cache:
                positions = {}
                for uid, binary in islice(source, len(ids)):
                    positions[uid] = (image_cache.tell(), len(binary))
                    image_cache.write(binary)
                if positions.keys() != ids:
                    raise ValueError("Candidate images do not match ViDoRe V3 corpus pages")
                for query in group:
                    images = []
                    for uid in rankings[query["qid"]]:
                        offset, length = positions[uid]
                        image_cache.seek(offset)
                        images.append(image_cache.read(length))
                    yield query, images
