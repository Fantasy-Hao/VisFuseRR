import math
from contextlib import contextmanager
from pathlib import Path

from utils import ROOT, local_options, open_rgb

from . import rerank, rerank_parser
from .model import DEFAULT_RERANKER, RERANKERS


def main(argv=None):
    cli = rerank_parser("Rerank candidates with a local pointwise model")
    cli.add_argument("--reranker-model-name", choices=RERANKERS, default=DEFAULT_RERANKER)
    cli.add_argument("--reranker-model", type=Path, help="Override the local reranker directory")
    local_options(cli)
    args = cli.parse_args(argv)
    path = args.reranker_model or ROOT / "model" / args.reranker_model_name

    @contextmanager
    def open_ranker(_output_dir):
        print(f"Loading pointwise model {args.reranker_model_name}", flush=True)
        model = RERANKERS[args.reranker_model_name](path, args.device)

        def rank_pages(query, page_ids, images, _texts):
            scores = []
            for start in range(0, len(images), args.batch_size):
                batch = [open_rgb(image) for image in images[start:start + args.batch_size]]
                batch_scores = model.score_batch(query, batch)
                if len(batch_scores) != len(batch) or not all(map(math.isfinite, batch_scores)):
                    raise ValueError("Pointwise model returned invalid batch scores")
                scores.extend(batch_scores)
            pairs = sorted(zip(page_ids, scores, strict=True), key=lambda pair: -pair[1])
            return {
                "ranking": [uid for uid, _ in pairs],
                "scores": [{"page_uid": uid, "score": score} for uid, score in pairs],
            }

        yield rank_pages

    try:
        rerank(args, open_ranker, method="pointwise", model_name=args.reranker_model_name)
    except (FileNotFoundError, ValueError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
