import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from common.dataset import open_dataset
from common.utils import embedding_folder, parallel_map, parser, positive_int, read_jsonl, retrieval_dir, write_json
from reranker.metrics import evaluate, evaluate_localization
from retriever.model import DEFAULT_EMBEDDER, EMBEDDERS


def rerank_parser(description):
    cli = parser(description)
    cli.add_argument("--embedding-model-name", choices=EMBEDDERS, default=DEFAULT_EMBEDDER)
    cli.add_argument("--topk", type=positive_int, default=10)
    cli.add_argument("--candidates", type=Path, help="Override the retrieval candidate JSONL file")
    return cli


def candidate_rankings(dataset, queries, path, topk):
    rows = read_jsonl(path)
    candidates = {row["qid"]: row for row in rows}
    if len(candidates) != len(rows):
        raise ValueError("Duplicate query IDs in retrieval output")

    rankings = {}
    for query in queries:
        qid = query["qid"]
        row = candidates.get(qid)
        if row is None or any(
            row.get(key) != value for key, value in query.items()
            if key not in ("gt_pages", "gt_boxes") and (key != "content_type" or key in row)
        ):
            raise ValueError(f"Retrieval candidates do not match query {qid}. Run retrieval first.")
        ids = [candidate["page_uid"] for candidate in row["candidates"][:topk]]
        allowed = dataset.candidate_ids(query)
        if len(ids) != min(topk, len(allowed)):
            raise ValueError("Too few candidates. Increase retrieval --topk or reduce rerank --topk.")
        if len(set(ids)) != len(ids) or any(type(uid) is not int or uid not in allowed for uid in ids):
            raise ValueError(f"Invalid candidate pages for query {qid}")
        rankings[qid] = ids
    return rankings


def rerank(
    args, rank_pages, *, method, model_name, variant=None,
    include_ocr=False, include_images=True, workers=1,
):
    dataset = open_dataset(args.dataset_name, args.dataset)
    localize = method == "listwise"
    queries, texts = dataset.load(args.limit, include_ocr=include_ocr, include_layouts=localize)
    path = args.candidates or retrieval_dir(args) / "retrieval" / "topk_candidates.jsonl"
    if not path.exists() and args.candidates is None and args.embedding_model_name == DEFAULT_EMBEDDER:
        legacy_path = args.output / "retriever" / "topk_candidates.jsonl"
        if legacy_path.is_file():
            path = legacy_path
    before = candidate_rankings(dataset, queries, path, args.topk)

    def rank_query(item):
        query, images = item
        qid = query["qid"]
        ids = before[qid]
        page_texts = [texts[uid] for uid in ids] if include_ocr else None
        try:
            result = rank_pages(query["query"], ids, images, page_texts)
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"Reranking failed for query {qid}: {error}") from error
        return {"qid": qid, **result}

    def collect(rows):
        results = []
        for result in rows:
            results.append(result)
            if len(results) % 20 == 0 or len(results) == len(queries):
                print(f"Reranked {len(results)}/{len(queries)} queries", flush=True)
        return results

    inputs = dataset.rerank_inputs(queries, before) if include_images else ((query, None) for query in queries)
    if workers == 1:
        results = collect(map(rank_query, inputs))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = collect(parallel_map(executor, rank_query, inputs, workers))

    results.sort(key=lambda row: row["qid"])
    metrics = evaluate(queries, before, results)
    localization = evaluate_localization(queries, results) if localize else None
    model_folder = quote(model_name, safe="")
    if model_folder in (".", ".."):
        raise ValueError("Invalid reranker model name")
    dest = args.output / "reranker" / args.dataset_name / embedding_folder(args.embedding_model_name)
    dest = dest / "rerank" / method / model_folder
    if variant is not None:
        dest = dest / variant
    write_json(dest / "rerank_scores.jsonl", results, jsonl=True)
    write_json(dest / "metrics_rerank.json", metrics)
    if localization is not None:
        write_json(dest / "metrics_localization.json", localization)
        print(f"Evaluated evidence boxes on {localization['n_pages']} relevant candidate pages")
    failures = sum("api_error" in row for row in results)
    if failures:
        print(f"API fallback to retrieval order: {failures}/{len(results)} queries")
    print(f"Recall@5: {metrics['before']['micro']['Recall@5']:.4f} -> {metrics['after']['micro']['Recall@5']:.4f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    method = argv.pop(0) if argv and argv[0] in ("pointwise", "listwise") else "pointwise"
    if method == "listwise":
        from reranker.listwise import main
    else:
        from reranker.pointwise import main
    main(argv)
