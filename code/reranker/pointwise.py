import math
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

from common.utils import ROOT, local_options, open_rgb, parallel_map, positive_int
from reranker.model import DEFAULT_RERANKER, RERANKERS
from reranker.rerank import rerank, rerank_parser

def main(argv=None):
    cli = rerank_parser("Rerank candidates with a local pointwise model")
    cli.add_argument("--reranker-model-name", choices=RERANKERS, default=DEFAULT_RERANKER)
    cli.add_argument("--reranker-model", type=Path, help="Override the local reranker directory")
    local_options(cli)
    cli.add_argument("--workers", type=positive_int, default=4, help="Parallel page image decoders")
    args = cli.parse_args(argv)
    model = None

    def rank_pages(query, page_ids, images, _texts):
        nonlocal model
        if model is None:
            path = args.reranker_model or ROOT / "model" / args.reranker_model_name
            model = RERANKERS[args.reranker_model_name](path, args.device)
        scores = []
        decoded = parallel_map(executor, open_rgb, images, max(args.batch_size, 2 * args.workers))
        while batch := list(islice(decoded, args.batch_size)):
            batch_scores = model.score_batch(query, batch)
            if len(batch_scores) != len(batch) or not all(map(math.isfinite, batch_scores)):
                raise ValueError("Invalid pointwise scores")
            scores.extend(batch_scores)
        order = sorted(range(len(page_ids)), key=scores.__getitem__, reverse=True)
        return {
            "ranking": [page_ids[index] for index in order],
            "scores": [{"page_uid": page_ids[index], "score": scores[index]} for index in order],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rerank(args, rank_pages, method="pointwise", model_name=args.reranker_model_name)


if __name__ == "__main__":
    main()
