from itertools import islice
from pathlib import Path

import numpy as np

from dataset import dataset_from_args, parser
from retriever.model import DEFAULT_EMBEDDER, EMBEDDERS
from utils import ROOT, atomic_output, dataset_folder, embeddings_path, group_candidates, local_options, open_rgb


def archive_selection(args):
    directory = args.dataset or ROOT / "data" / args.dataset_name
    return [dataset_folder(args).as_posix(), str(directory.resolve()), args.embedding_model_name]


def encode_items(model, items, batch_size, role, label, count):
    items = iter(items)
    batches = []
    processed = 0
    while batch := list(islice(items, batch_size)):
        vectors = np.asarray(model.encode_batch(batch, role), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(batch) or not np.isfinite(vectors).all():
            raise ValueError(f"{label}: model returned invalid embeddings")
        batches.append(vectors)
        processed += len(batch)
        if len(batches) % 100 == 0 or processed == count:
            print(f"{label}: {processed}/{count}", flush=True)
    if processed != count:
        raise ValueError(f"{label}: expected {count} embeddings, got {processed}")
    return np.concatenate(batches)


def encode(args):
    dataset = dataset_from_args(args)
    queries, texts = dataset.load(args.limit, include_ocr=True)
    query_ids = [query["qid"] for query in queries]
    page_ids = sorted({uid for group in group_candidates(queries) for uid in dataset.candidate_ids(group[0])})
    model_path = args.embedding_model or ROOT / "model" / args.embedding_model_name
    embedder = EMBEDDERS[args.embedding_model_name](model_path, args.device)
    query_inputs = ({"type": "text", "text": query["query"] or "NULL"} for query in queries)
    text_inputs = (
        {"type": "text", "text": texts[uid].strip() or "NULL"}
        for uid in page_ids
    )
    query_vectors = encode_items(embedder, query_inputs, args.batch_size, "query", "Queries", len(queries))
    text_vectors = encode_items(embedder, text_inputs, args.batch_size, "passage", "Page OCR", len(page_ids))
    image_inputs = ({"type": "image", "image": open_rgb(binary)} for _, binary in dataset.images(page_ids))
    image_vectors = encode_items(embedder, image_inputs, args.batch_size, "image", "Page images", len(page_ids))

    path = embeddings_path(args)
    with atomic_output(path, "wb") as stream:
        np.savez(
            stream, selection=archive_selection(args), query_ids=query_ids, page_ids=page_ids,
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
