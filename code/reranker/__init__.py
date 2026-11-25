"""Shared candidate loading, execution and evaluation for page reranking."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from utils import (
    experiment_folder, parallel_map, positive_int, read_jsonl, report_progress,
    rerank_document_inputs, results_dir, retrieval_dir, write_json,
)


def rerank_parser(description):
    from dataset import parser
    from retriever.model import DEFAULT_EMBEDDER, EMBEDDERS

    cli = parser(description)
    cli.add_argument("--embedding-model-name", choices=EMBEDDERS, default=DEFAULT_EMBEDDER)
    cli.add_argument("--topk", type=positive_int, default=10)
    cli.add_argument("--candidates", type=Path, help="Override the retrieval candidate JSONL file")
    return cli


def _candidate_rankings(dataset, queries, path, topk):
    candidates = {}
    for position, row in enumerate(read_jsonl(path), 1):
        if not isinstance(row, dict) or type(row.get("qid")) is not int:
            raise ValueError(f"Expected an integer qid on candidate record {position} in {path}")
        qid = row["qid"]
        if qid in candidates:
            raise ValueError(f"Duplicate query identifier {qid} in {path}")
        candidates[qid] = row
    rankings = {}
    for query in queries:
        qid = query["qid"]
        row = candidates.get(qid)
        if row is None or any(row.get(key) != query[key] for key in ("query", "cand_span")):
            raise ValueError(f"Retrieval candidates do not match query {qid}. Run retrieval first.")
        if any(key in row and row[key] != query.get(key) for key in ("source_qid", "doc_id")):
            raise ValueError(f"Retrieval candidates belong to a different document or query: {qid}")
        entries = row.get("candidates")
        if not isinstance(entries, list):
            raise ValueError(f"Expected a candidates array for query {qid} in {path}")
        ids = [entry.get("page_uid") if isinstance(entry, dict) else None for entry in entries[:topk]]
        allowed = dataset.candidate_ids(query)
        if len(ids) != min(topk, len(allowed)):
            raise ValueError("Too few candidates. Increase retrieval --topk or reduce rerank --topk.")
        if any(type(uid) is not int or uid not in allowed for uid in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid candidate pages for query {qid}")
        rankings[qid] = ids
    return rankings


def rerank(args, rerank_pages, *, method, model_name, workers=1):
    from dataset import dataset_from_args
    from .metrics import evaluate, evaluate_localization

    path = args.candidates or retrieval_dir(args) / "retrieval" / "topk_candidates.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing retrieval candidates at {path}. Run bash run.sh retrieve first.")
    dest = args.output / "reranker" / experiment_folder(args) / "rerank" / method / quote(model_name, safe="")
    metrics_dest = results_dir(args, dest)
    dataset = dataset_from_args(args)
    listwise = method == "listwise"
    queries, texts = dataset.load(args.limit, include_ocr=listwise, include_layouts=listwise)
    if not queries:
        raise ValueError("No queries selected for reranking")
    print(f"Loading retrieval candidates from {path}", flush=True)
    before = _candidate_rankings(dataset, queries, path, args.topk)

    def rank_query(item):
        query, images = item
        qid = query["qid"]
        ids = before[qid]
        page_texts = [texts[uid] for uid in ids] if listwise else None
        return {"qid": qid, **rerank_pages(query["query"], ids, images, page_texts)}

    label = f"{method.capitalize()} reranking queries"
    count = len(queries)
    report_progress(label, 0, count, every=10)
    inputs = rerank_document_inputs(dataset.images, queries, before)
    results = []

    def collect(rows):
        for completed, result in enumerate(rows, 1):
            results.append(result)
            report_progress(label, completed, count, every=10)

    if workers == 1:
        collect(map(rank_query, inputs))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            collect(parallel_map(executor, rank_query, inputs, workers))

    results.sort(key=lambda row: row["qid"])
    metrics = evaluate(queries, before, results)
    failures = metrics["run"]["fallback_queries"]
    write_json(dest / "rerank_scores.jsonl", results, jsonl=True)
    write_json(metrics_dest / "metrics_rerank.json", metrics)
    if listwise:
        write_json(metrics_dest / "metrics_localization.json", evaluate_localization(queries, results))
    print(f"Saved reranking metrics to {metrics_dest}", flush=True)
    if failures:
        print(f"API fallback to retrieval order: {failures}/{len(results)} queries", flush=True)
    if failures == len(results):
        raise RuntimeError("All reranking requests failed. See api_error in rerank_scores.jsonl.")
    print(f"Recall@5: {metrics['before']['micro']['Recall@5']:.4f} -> {metrics['after']['micro']['Recall@5']:.4f}", flush=True)
