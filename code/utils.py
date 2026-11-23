import argparse
import json
from collections import deque
from contextlib import contextmanager
from itertools import groupby, islice
from io import BytesIO
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = {"shortest_edge": 4 * 32 * 32, "longest_edge": 1800 * 32 * 32}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def local_options(cli):
    cli.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    cli.add_argument("--batch-size", type=positive_int, default=2)


def local_device(device):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    return torch.device(device), dtype


def open_rgb(binary):
    with Image.open(BytesIO(binary)) as image:
        return image.convert("RGB")


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


def candidate_span(query):
    start, end = query["cand_span"]
    return range(start, end + 1)


def group_candidates(queries):
    by_span = sorted(queries, key=lambda query: query["cand_span"])
    for _, group in groupby(by_span, key=lambda query: query["cand_span"]):
        yield list(group)


def rerank_document_inputs(images, queries, rankings):
    """Read each selected page once, retaining the images of one document at a time."""
    source = images({uid for ids in rankings.values() for uid in ids})
    for group in group_candidates(queries):
        ids = {uid for query in group for uid in rankings[query["qid"]]}
        by_id = dict(islice(source, len(ids)))
        if by_id.keys() != ids:
            raise ValueError("Candidate images do not match document pages")
        for query in group:
            yield query, [by_id[uid] for uid in rankings[query["qid"]]]


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@contextmanager
def atomic_output(path, mode="w", **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode=mode, dir=path.parent, delete=False, **kwargs) as stream:
        pending = Path(stream.name)
        try:
            yield stream
            stream.close()
            pending.replace(path)
        finally:
            stream.close()
            pending.unlink(missing_ok=True)


def write_json(path, value, *, jsonl=False):
    with atomic_output(path, encoding="utf-8") as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


def parallel_map(executor, function, items, buffer_size):
    """Preserve input order and limit the number of items retained in memory."""
    items = iter(items)
    pending = deque(executor.submit(function, item) for item in islice(items, buffer_size))
    while pending:
        yield pending.popleft().result()
        pending.extend(executor.submit(function, item) for item in islice(items, 1))


def embeddings_path(args):
    return args.embeddings or retrieval_dir(args) / "encode" / "embeddings.npz"


def embedding_folder(model_name):
    return "Qwen3VLEmb8B" if model_name == "Qwen3-VL-Embedding-8B" else model_name


def retrieval_dir(args):
    return args.output / "retriever" / dataset_folder(args) / embedding_folder(args.embedding_model_name)


def dataset_folder(args):
    folder = Path(args.dataset_name)
    if args.dataset_name == "docscope":
        return folder / (args.split or "test")
    if args.dataset_name == "vidore_v3":
        return folder / (args.subset or "all")
    return folder
