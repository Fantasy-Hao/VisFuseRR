import json
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from common.dataset import open_dataset
from common.utils import (
    EMBEDDING_SCHEMA, ROOT, atomic_output, embeddings_path, file_stamp,
    local_options, open_rgb, parallel_map, parser, positive_int,
)
from retriever.model import DEFAULT_EMBEDDER, EMBEDDERS


def encode_items(model, items, batch_size, role, label, count):
    items = iter(items)
    batches = []
    processed = 0
    while batch := list(islice(items, batch_size)):
        vectors = np.asarray(model.encode_batch(batch, role), dtype=np.float32)
        if (
            vectors.ndim != 2 or len(vectors) != len(batch)
            or not np.isfinite(vectors).all() or not np.any(vectors, axis=1).all()
        ):
            raise ValueError(f"{label}: model returned invalid embeddings")
        batches.append(vectors)
        processed += len(batch)
        if len(batches) % 100 == 0 or processed == count:
            print(f"{label}: {processed}/{count}", flush=True)
    if processed != count:
        raise ValueError(f"{label}: expected {count} embeddings, got {processed}")
    return np.concatenate(batches)


def encode(args):
    dataset = open_dataset(args.dataset_name, args.dataset)
    queries, texts = dataset.load(args.limit, include_ocr=True)
    query_ids = [query["qid"] for query in queries]
    page_ids = sorted({uid for query in queries for uid in dataset.candidate_ids(query)})
    path = embeddings_path(args)
    model_path = args.embedding_model or ROOT / "model" / args.embedding_model_name
    model_class = EMBEDDERS[args.embedding_model_name]
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local embedding model directory not found: {model_path}")
    model_files = sorted(
        file for file in model_path.rglob("*")
        if file.is_file() and file.suffix in {".json", ".jinja", ".safetensors", ".txt", ".py"}
    )
    metadata = json.dumps({
        "schema": EMBEDDING_SCHEMA,
        "dataset": dataset.signature(),
        "model": [file_stamp(file) for file in model_files],
        "model_name": args.embedding_model_name,
        "encoding": model_class.encoding,
        "device": args.device,
    }, sort_keys=True)
    if path.is_file() and not args.force:
        try:
            with np.load(path, allow_pickle=False) as saved:
                saved_metadata = json.loads(saved["metadata"].item())
                saved_metadata.setdefault("schema", EMBEDDING_SCHEMA)
                if (
                    json.dumps(saved_metadata, sort_keys=True) == metadata
                    and {"queries", "texts", "images"}.issubset(saved.files)
                    and np.isin(query_ids, saved["query_ids"]).all()
                    and np.isin(page_ids, saved["page_ids"]).all()
                ):
                    print(f"Reusing embeddings: {path}", flush=True)
                    return
        except (OSError, ValueError, KeyError, TypeError, AttributeError, BadZipFile):
            pass

    embedder = model_class(model_path, args.device)
    query_inputs = ({"type": "text", "text": query["query"] or "NULL"} for query in queries)
    text_inputs = (
        {"type": "text", "text": texts[uid].strip() or "NULL"}
        for uid in page_ids
    )
    query_vectors = encode_items(embedder, query_inputs, args.batch_size, "query", "Queries", len(queries))
    text_vectors = encode_items(embedder, text_inputs, args.batch_size, "passage", "Page OCR", len(page_ids))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        binaries = (binary for _, binary in dataset.images(page_ids))
        images = parallel_map(executor, open_rgb, binaries, max(args.batch_size, 2 * args.workers))
        image_inputs = ({"type": "image", "image": image} for image in images)
        image_vectors = encode_items(embedder, image_inputs, args.batch_size, "image", "Page images", len(page_ids))
    if query_vectors.shape[1] != text_vectors.shape[1] or text_vectors.shape != image_vectors.shape:
        raise ValueError("Query, OCR, and image embeddings must have the same dimension")

    with atomic_output(path, "wb") as stream:
        np.savez(
            stream, metadata=metadata, query_ids=query_ids, page_ids=page_ids,
            queries=query_vectors, texts=text_vectors, images=image_vectors,
        )
    print(f"Saved embeddings: {path}", flush=True)


def main(argv=None):
    cli = parser("Encode dataset queries, OCR and page images with a local embedding model")
    cli.add_argument("--embedding-model-name", choices=EMBEDDERS, default=DEFAULT_EMBEDDER)
    cli.add_argument("--embedding-model", type=Path, help="Override the local model directory")
    cli.add_argument("--embeddings", type=Path, help="Override the output embedding archive")
    cli.add_argument("--force", action="store_true", help="Rebuild an existing archive")
    cli.add_argument("--workers", type=positive_int, default=4, help="Parallel page image decoders")
    local_options(cli)
    encode(cli.parse_args(argv))


if __name__ == "__main__":
    main()
