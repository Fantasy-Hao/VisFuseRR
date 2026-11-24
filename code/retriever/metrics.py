import math
from collections import defaultdict
from statistics import mean

KS = (1, 3, 5, 10)


def query_metrics(ranking, relevant_pages, ks=KS):
    relevant = set(relevant_pages)
    if not relevant:
        raise ValueError("A query must have at least one ground-truth page")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Duplicate page identifiers in a ranking")

    metrics = {}
    for k in ks:
        retrieved = ranking[:k]
        hit_ranks = [rank for rank, uid in enumerate(retrieved, 1) if uid in relevant]
        ideal_hits = min(k, len(relevant))
        dcg = sum(1 / math.log2(rank + 1) for rank in hit_ranks)
        ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        precision_sum = sum(hits / rank for hits, rank in enumerate(hit_ranks, 1))
        metrics.update({
            f"Recall@{k}": len(hit_ranks) / len(relevant),
            f"RecallAll@{k}": int(len(hit_ranks) == len(relevant)),
            f"Hit@{k}": int(bool(hit_ranks)),
            f"Precision@{k}": len(hit_ranks) / len(retrieved) if retrieved else 0,
            f"NDCG@{k}": dcg / ideal_dcg,
            f"MAP@{k}": precision_sum / ideal_hits,
            f"MRR@{k}": 1 / hit_ranks[0] if hit_ranks else 0,
        })
    return metrics


def evaluate(queries, rankings, ks=KS):
    rows = []
    domains = defaultdict(list)
    for query in queries:
        qid, group = query["qid"], query["group"]
        metrics = query_metrics(rankings[qid], query["gt_pages"], ks)
        rows.append({"qid": qid, "group": group, **metrics})
        domains[group].append(metrics)
    if not rows:
        raise ValueError("No evaluation queries")
    keys = [key for key in rows[0] if "@" in key]

    def averages(items):
        return {key: mean(item[key] for item in items) for key in keys}

    by_domain = {
        group: {"n": len(items), **averages(items)} for group, items in sorted(domains.items())
    }
    return {
        "n": len(rows), "micro": averages(rows), "macro": averages(by_domain.values()),
        "by_domain": by_domain, "per_query": rows,
    }
