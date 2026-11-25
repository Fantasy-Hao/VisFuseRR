import argparse
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from dataset import dataset_from_args, parser
from utils import (
    archive_selection, dataset_folder, embeddings_path, group_candidates, positive_int,
    query_key, report_progress, results_dir, retrieval_dir, write_json,
)

from .fusion import IMAGE_WEIGHT, fuse, fusion_inputs
from .metrics import evaluate
from .model import DEFAULT_EMBEDDER, EMBEDDERS


def rank(scores, page_ids, topk):
    return np.lexsort((page_ids, -scores))[:topk]


def image_weight(value):
    weight = float(value)
    if not 0 <= weight <= 1:
        raise argparse.ArgumentTypeError("Image weight must be between 0 and 1")
    return weight


def grid_search(rows):
    """Tune the global image weight by mean per-query Recall@5."""
    recalls = {}

    def best(steps):
        for step in steps:
            if step not in recalls:
                recalls[step] = float(np.mean([
                    relevant[rank(fuse(text, image, step / 100), page_ids, 5)].sum() / count
                    for page_ids, text, image, relevant, count in rows
                ]))
        default_step = round(100 * IMAGE_WEIGHT)
        return max(recalls, key=lambda step: (recalls[step], -abs(step - default_step), -step))

    coarse = best(range(0, 101, 5))
    optimum = best(range(max(0, coarse - 5), min(100, coarse + 5) + 1))
    return optimum / 100, [
        {"alpha": step / 100, "Recall@5": recalls[step]} for step in sorted(recalls)
    ]


def load_embeddings(args, queries):
    path = embeddings_path(args)
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding archive at {path}. Run bash run.sh encode first.")
    archive = np.load(path, allow_pickle=False)
    if isinstance(archive, np.ndarray):
        raise ValueError(f"Expected a .npz embedding archive at {path}. Run bash run.sh encode first.")
    with archive as saved:
        required = {"selection", "query_ids", "page_ids", "queries", "texts", "images"}
        if missing := required.difference(saved.files):
            raise ValueError(f"Missing embedding archive fields at {path}: {', '.join(sorted(missing))}")
        if saved["selection"].tolist() != archive_selection(args):
            raise ValueError(f"Run bash run.sh encode for the selected dataset and model to rebuild {path}")
        query_ids, page_ids = saved["query_ids"], saved["page_ids"]
        vectors, texts, images = saved["queries"], saved["texts"], saved["images"]
        if (
            any(ids.ndim != 1 or ids.dtype.kind not in "iu" for ids in (query_ids, page_ids))
            or any(matrix.ndim != 2 or matrix.dtype.kind != "f" for matrix in (vectors, texts, images))
            or texts.shape != images.shape
            or vectors.shape[1] != texts.shape[1] or vectors.shape[1] == 0
            or len(query_ids) != len(vectors) or len(page_ids) != len(texts)
        ):
            raise ValueError(f"Invalid embedding archive dimensions or page/query IDs: {path}")
        query_vectors = dict(zip(query_ids.tolist(), vectors))
        page_index = {uid: index for index, uid in enumerate(page_ids.tolist())}
        if len(query_vectors) != len(query_ids) or len(page_index) != len(page_ids):
            raise ValueError(f"Duplicate IDs in embedding archive: {path}")
        if "query_keys" in saved:
            query_keys = saved["query_keys"]
            if query_keys.shape != query_ids.shape:
                raise ValueError(f"Invalid query metadata in embedding archive: {path}")
            keys = dict(zip(query_ids.tolist(), query_keys.tolist()))
            for query in queries:
                if keys.get(query["qid"]) != query_key(query):
                    raise ValueError(f"Query {query['qid']} is missing or changed. Rebuild {path} with encode.")
    return query_vectors, page_index, texts, images


def score_queries(dataset, queries, embeddings, fusion):
    query_vectors, page_index, text_vectors, image_vectors = embeddings
    for group in group_candidates(queries):
        pids = np.asarray(dataset.candidate_ids(group[0]))
        try:
            indices = [page_index[uid] for uid in pids]
            vectors = np.stack([query_vectors[query["qid"]] for query in group])
        except KeyError as error:
            raise ValueError("The archive does not cover this selection. Re-run encode for these queries.") from error
        page_texts, page_images = text_vectors[indices], image_vectors[indices]
        batch_size = 64
        for start in range(0, len(group), batch_size):
            end = start + batch_size
            text_scores = vectors[start:end] @ page_texts.T
            image_scores = vectors[start:end] @ page_images.T
            for query, text, image in zip(group[start:end], text_scores, image_scores, strict=True):
                if not (np.isfinite(text).all() and np.isfinite(image).all()):
                    raise ValueError(f"Non-finite retrieval scores for query {query['qid']}. Check the embeddings.")
                text_fusion, image_fusion = fusion_inputs(text, image, fusion)
                yield query, pids, text, image, text_fusion, image_fusion


def search(args):
    dest = retrieval_dir(args) / "retrieval"
    metrics_dest = results_dir(args, dest)
    print(f"Loading {dataset_folder(args)} queries", flush=True)
    dataset = dataset_from_args(args)
    queries, _ = dataset.load(args.limit)
    if not queries:
        raise ValueError("No queries selected for retrieval")
    count = len(queries)
    print(f"Loading embeddings for {count} queries from {embeddings_path(args)}", flush=True)
    embeddings = load_embeddings(args, queries)

    alpha = args.alpha
    scored = score_queries(dataset, queries, embeddings, args.fusion)
    if args.grid_search:
        print("Tuning the image weight by Recall@5", flush=True)
        scored = list(scored)
        grid_rows = [
            (pids, text, image, np.isin(pids, query["gt_pages"]), len(query["gt_pages"]))
            for query, pids, _, _, text, image in scored
        ]
        alpha, trials = grid_search(grid_rows)
        write_json(metrics_dest / "gridsearch.json", {"fusion": args.fusion, "best_alpha": alpha, "trials": trials})
        print(f"Grid search: best image weight={alpha:.2f}", flush=True)

    candidates, rankings = {}, {}
    label = "Retrieving queries"
    report_progress(label, 0, count, every=100)
    for completed, (query, pids, text, image, text_fusion, image_fusion) in enumerate(scored, 1):
        scores = fuse(text_fusion, image_fusion, alpha)
        order = rank(scores, pids, args.topk)
        selected = [
            {
                "page_uid": int(pids[i]), "rank": position, "score": float(scores[i]),
                "text_score": float(text[i]), "image_score": float(image[i]),
            }
            for position, i in enumerate(order, 1)
        ]
        candidates[query["qid"]] = {
            **{key: value for key, value in query.items() if not key.startswith("gt_")},
            "alpha": alpha, "candidates": selected,
        }
        rankings[query["qid"]] = [candidate["page_uid"] for candidate in selected]
        report_progress(label, completed, count, every=100)

    print(f"Writing retrieval candidates to {dest}", flush=True)
    write_json(dest / "topk_candidates.jsonl", (candidates[query["qid"]] for query in queries), jsonl=True)
    metrics = evaluate(queries, rankings)
    metrics["retrieval"] = {"fusion": args.fusion, "alpha": alpha, "topk": args.topk, "grid_search": args.grid_search}
    write_json(metrics_dest / "metrics_retrieval.json", metrics)
    print(f"Saved retrieval metrics to {metrics_dest}", flush=True)
    print(f"Retrieved {count} queries. Recall@5={metrics['micro']['Recall@5']:.4f}", flush=True)


def main(argv=None):
    cli = parser("Retrieve pages from a saved embedding archive")
    cli.add_argument("--embedding-model-name", choices=EMBEDDERS, default=DEFAULT_EMBEDDER)
    cli.add_argument("--embeddings", type=Path, help="Override the input embedding archive")
    cli.add_argument("--fusion", choices=("linear", "sigmoid-zscore"), default="linear")
    weights = cli.add_mutually_exclusive_group()
    weights.add_argument("--alpha", type=image_weight, default=IMAGE_WEIGHT, help="Fixed image weight in [0, 1] (default: 0.60)")
    weights.add_argument("--grid-search", action="store_true", help="Tune the global image weight on Recall@5")
    cli.add_argument("--topk", type=positive_int, default=50)
    args = cli.parse_args(argv)
    try:
        search(args)
    except (BadZipFile, EOFError) as error:
        cli.error(f"Cannot read embedding archive at {embeddings_path(args)}: {error}")
    except (FileNotFoundError, ValueError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
