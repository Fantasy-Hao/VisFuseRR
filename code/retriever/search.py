import argparse
from pathlib import Path

import numpy as np

from dataset import dataset_from_args, parser
from retriever.encode import archive_selection
from retriever.fusion import IMAGE_WEIGHT, fuse, fusion_inputs
from retriever.metrics import evaluate
from retriever.model import DEFAULT_EMBEDDER, EMBEDDERS
from utils import embeddings_path, group_candidates, positive_int, retrieval_dir, write_json


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


def load_embeddings(args):
    path = embeddings_path(args)
    with np.load(path, allow_pickle=False) as saved:
        if "selection" not in saved or saved["selection"].tolist() != archive_selection(args):
            raise ValueError(f"Run retriever.encode for the selected dataset and model to rebuild {path}")
        queries = dict(zip(saved["query_ids"].tolist(), saved["queries"], strict=True))
        page_index = {uid: index for index, uid in enumerate(saved["page_ids"].tolist())}
        return queries, page_index, saved["texts"], saved["images"]


def score_queries(dataset, queries, embeddings, fusion):
    query_vectors, page_index, text_vectors, image_vectors = embeddings
    for group in group_candidates(queries):
        pids = np.asarray(dataset.candidate_ids(group[0]))
        indices = [page_index[uid] for uid in pids]
        vectors = np.stack([query_vectors[query["qid"]] for query in group])
        page_texts, page_images = text_vectors[indices], image_vectors[indices]
        batch_size = 64
        for start in range(0, len(group), batch_size):
            end = start + batch_size
            text_scores = vectors[start:end] @ page_texts.T
            image_scores = vectors[start:end] @ page_images.T
            for query, text, image in zip(group[start:end], text_scores, image_scores):
                text_fusion, image_fusion = fusion_inputs(text, image, fusion)
                yield query, pids, text, image, text_fusion, image_fusion


def search(args):
    dataset = dataset_from_args(args)
    queries, _ = dataset.load(args.limit)
    embeddings = load_embeddings(args)

    dest = retrieval_dir(args) / "retrieval"
    alpha = args.alpha
    if args.grid_search:
        grid_rows = [
            (pids, text, image, np.isin(pids, query["gt_pages"]), len(query["gt_pages"]))
            for query, pids, _, _, text, image in score_queries(dataset, queries, embeddings, args.fusion)
        ]
        alpha, trials = grid_search(grid_rows)
        del grid_rows
        write_json(dest / "gridsearch.json", {"fusion": args.fusion, "best_alpha": alpha, "trials": trials})
        print(f"Grid search: best image weight={alpha:.2f}")

    candidates, rankings = {}, {}
    for query, pids, text, image, text_fusion, image_fusion in score_queries(dataset, queries, embeddings, args.fusion):
        scores = fuse(text_fusion, image_fusion, alpha)
        order = rank(scores, pids, args.topk)
        selected = [
            {
                "page_uid": int(pids[i]), "rank": rank, "score": float(scores[i]),
                "text_score": float(text[i]), "image_score": float(image[i]),
            }
            for rank, i in enumerate(order, 1)
        ]
        candidates[query["qid"]] = {
            **{key: value for key, value in query.items() if not key.startswith("gt_")},
            "alpha": alpha, "candidates": selected,
        }
        rankings[query["qid"]] = [candidate["page_uid"] for candidate in selected]

    write_json(dest / "topk_candidates.jsonl", (candidates[query["qid"]] for query in queries), jsonl=True)
    metrics = evaluate(queries, rankings)
    write_json(dest / "metrics_retrieval.json", metrics)
    print(f"Retrieved {len(queries)} queries. Recall@5={metrics['micro']['Recall@5']:.4f}")


def main(argv=None):
    cli = parser("Retrieve pages from a saved embedding archive")
    cli.add_argument("--embedding-model-name", choices=EMBEDDERS, default=DEFAULT_EMBEDDER)
    cli.add_argument("--embeddings", type=Path, help="Override the input embedding archive")
    cli.add_argument("--fusion", choices=("linear", "sigmoid-zscore"), default="linear")
    weights = cli.add_mutually_exclusive_group()
    weights.add_argument("--alpha", type=image_weight, default=IMAGE_WEIGHT, help="Fixed image weight in [0, 1] (default: 0.60)")
    weights.add_argument("--grid-search", action="store_true", help="Tune the global image weight on Recall@5")
    cli.add_argument("--topk", type=positive_int, default=50)
    search(cli.parse_args(argv))


if __name__ == "__main__":
    main()
