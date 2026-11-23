import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from dataset import dataset_from_args, parser
from utils import (
    dataset_folder, embedding_folder, parallel_map,
    positive_int, read_jsonl, retrieval_dir, write_json,
)
from reranker.metrics import evaluate, evaluate_localization
from retriever.model import DEFAULT_EMBEDDER, EMBEDDERS


def rerank_parser(description):
    cli = parser(description)
    cli.add_argument("--embedding-model-name", choices=EMBEDDERS, default=DEFAULT_EMBEDDER)
    cli.add_argument("--topk", type=positive_int, default=10)
    cli.add_argument("--candidates", type=Path, help="Override the retrieval candidate JSONL file")
    return cli


def candidate_rankings(dataset, queries, path, topk):
    candidates = {row["qid"]: row for row in read_jsonl(path)}
    rankings = {}
    for query in queries:
        qid = query["qid"]
        row = candidates[qid]
        if row["query"] != query["query"] or row["cand_span"] != query["cand_span"]:
            raise ValueError(f"Retrieval candidates do not match query {qid}. Run retrieval first.")
        ids = [candidate["page_uid"] for candidate in row["candidates"][:topk]]
        allowed = dataset.candidate_ids(query)
        if len(ids) != min(topk, len(allowed)):
            raise ValueError("Too few candidates. Increase retrieval --topk or reduce rerank --topk.")
        if len(set(ids)) != len(ids) or any(type(uid) is not int or uid not in allowed for uid in ids):
            raise ValueError(f"Invalid candidate pages for query {qid}")
        rankings[qid] = ids
    return rankings


def rerank(args, rank_pages, *, method, model_name, workers=1):
    dataset = dataset_from_args(args)
    listwise = method == "listwise"
    queries, texts = dataset.load(args.limit, include_ocr=listwise, include_layouts=listwise)
    path = args.candidates or retrieval_dir(args) / "retrieval" / "topk_candidates.jsonl"
    before = candidate_rankings(dataset, queries, path, args.topk)

    def rank_query(item):
        query, images = item
        qid = query["qid"]
        ids = before[qid]
        page_texts = [texts[uid] for uid in ids] if listwise else None
        result = rank_pages(query["query"], ids, images, page_texts)
        return {"qid": qid, **result}

    inputs = dataset.rerank_inputs(queries, before)
    if workers == 1:
        results = list(map(rank_query, inputs))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(parallel_map(executor, rank_query, inputs, workers))

    results.sort(key=lambda row: row["qid"])
    metrics = evaluate(queries, before, results)
    model_folder = quote(model_name, safe="")
    if model_folder in (".", ".."):
        raise ValueError("Invalid reranker model name")
    dest = args.output / "reranker" / dataset_folder(args) / embedding_folder(args.embedding_model_name)
    dest = dest / "rerank" / method / model_folder
    write_json(dest / "rerank_scores.jsonl", results, jsonl=True)
    write_json(dest / "metrics_rerank.json", metrics)
    if listwise:
        localization = evaluate_localization(queries, results)
        write_json(dest / "metrics_localization.json", localization)
        print(f"Evaluated evidence boxes on {localization['n_pages']} relevant candidate pages")
    failures = sum("api_error" in row for row in results)
    if failures:
        print(f"API fallback to retrieval order: {failures}/{len(results)} queries")
    if failures == len(results):
        raise RuntimeError("All reranking requests failed. See api_error in rerank_scores.jsonl.")
    print(f"Recall@5: {metrics['before']['micro']['Recall@5']:.4f} -> {metrics['after']['micro']['Recall@5']:.4f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    method = argv.pop(0) if argv and argv[0] in ("pointwise", "listwise") else "pointwise"
    if method == "listwise":
        from reranker.listwise import main
    else:
        from reranker.pointwise import main
    main(argv)
