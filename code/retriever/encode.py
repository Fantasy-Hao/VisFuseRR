from itertools import islice
from pathlib import Path

import numpy as np

from dataset import dataset_from_args, parser
from utils import (
    ROOT, archive_selection, atomic_output, dataset_folder, embeddings_path, group_candidates,
    local_options, open_rgb, query_key, report_progress,
)

from .model import DEFAULT_EMBEDDER, EMBEDDERS


def encode_items(model, items, batch_size, role, label, count):
    items = iter(items)
    encoded = None
    processed = 0
    report_progress(label, 0, count, every=batch_size * 10)
    while batch := list(islice(items, batch_size)):
        vectors = np.asarray(model.encode_batch(batch, role), dtype=np.float32)
        if (
            vectors.ndim != 2 or vectors.shape[0] != len(batch)
            or not np.isfinite(vectors).all() or not vectors.any(axis=1).all()
        ):
            raise ValueError(f"{label}: model returned invalid embeddings")
        if encoded is None:
            encoded = np.empty((count, vectors.shape[1]), dtype=np.float32)
        end = processed + len(batch)
        if end > count or vectors.shape[1] != encoded.shape[1]:
            raise ValueError(f"{label}: unexpected embedding count or dimension")
        encoded[processed:end] = vectors
        processed = end
        report_progress(label, processed, count, every=batch_size * 10)
    if encoded is None or processed != count:
        raise ValueError(f"{label}: expected {count} embeddings, got {processed}")
    return encoded


def image_inputs(dataset, page_ids):
    for expected, (uid, binary) in zip(page_ids, dataset.images(page_ids), strict=True):
        if expected != uid:
            raise ValueError(f"Expected image for page {expected}, received page {uid}")
        yield {"type": "image", "image": open_rgb(binary)}


def encode(args):
    print(f"Loading {dataset_folder(args)} queries and page text", flush=True)
    dataset = dataset_from_args(args)
    queries, texts = dataset.load(args.limit, include_ocr=True)
    if not queries:
        raise ValueError("No queries selected for encoding")
    query_ids = [query["qid"] for query in queries]
    page_ids = sorted({uid for group in group_candidates(queries) for uid in dataset.candidate_ids(group[0])})
    model_path = args.embedding_model or ROOT / "model" / args.embedding_model_name
    print(f"Loading embedder {args.embedding_model_name} for {len(queries)} queries and {len(page_ids)} pages", flush=True)
    embedder = EMBEDDERS[args.embedding_model_name](model_path, args.device)
    query_inputs = ({"type": "text", "text": query["query"] or "NULL"} for query in queries)
    text_inputs = (
        {"type": "text", "text": texts[uid].strip() or "NULL"}
        for uid in page_ids
    )
    query_vectors = encode_items(embedder, query_inputs, args.batch_size, "query", "Queries", len(queries))
    text_vectors = encode_items(embedder, text_inputs, args.batch_size, "passage", "Page text", len(page_ids))
    image_vectors = encode_items(
        embedder, image_inputs(dataset, page_ids), args.batch_size, "image", "Page images", len(page_ids),
    )
    if text_vectors.shape != image_vectors.shape or query_vectors.shape[1] != text_vectors.shape[1]:
        raise ValueError("Query, text and image embeddings have incompatible dimensions")

    path = embeddings_path(args)
    with atomic_output(path, "wb") as stream:
        np.savez(
            stream, selection=archive_selection(args), query_ids=query_ids, page_ids=page_ids,
            query_keys=[query_key(query) for query in queries],
            queries=query_vectors, texts=text_vectors, images=image_vectors,
        )
    print(f"Saved embeddings: {path}", flush=True)


def main(argv=None):
    cli = parser("Encode dataset queries, OCR and page images with a local embedding model")
    cli.add_argument("--embedding-model-name", choices=EMBEDDERS, default=DEFAULT_EMBEDDER)
    cli.add_argument("--embedding-model", type=Path, help="Override the local model directory")
    cli.add_argument("--embeddings", type=Path, help="Override the output embedding archive")
    local_options(cli)
    encode(cli.parse_args(argv))


if __name__ == "__main__":
    main()
