import json
import sqlite3
from contextlib import closing, contextmanager
from functools import partial
from threading import Lock

from utils import positive_int

from . import rerank, rerank_parser
from .model import Qwen38Listwise


class Checkpoint:
    def __init__(self, connection):
        self.connection = connection
        self.lock = Lock()
        self.reused = 0
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS completed (request_key TEXT PRIMARY KEY, result TEXT NOT NULL)"
            )

    def get(self, key):
        with self.lock:
            row = self.connection.execute(
                "SELECT result FROM completed WHERE request_key = ?", (key,),
            ).fetchone()
            if row is not None:
                self.reused += 1
                return json.loads(row[0])
        return None

    def save(self, key, result):
        value = json.dumps(result, ensure_ascii=False, allow_nan=False)
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO completed (request_key, result) VALUES (?, ?)", (key, value),
            )


def main(argv=None):
    cli = rerank_parser("Rerank candidates with locally served Qwen3.8-27B-FP8")
    cli.add_argument("--base-url", default=Qwen38Listwise.BASE_URL, help="Loopback vLLM endpoint ending in /v1")
    cli.add_argument("--workers", type=positive_int, default=2, help="Concurrent local requests (default: 2)")
    cli.add_argument(
        "--read-timeout", type=positive_int, default=Qwen38Listwise.READ_TIMEOUT,
        help=f"Seconds to wait for response data (default: {Qwen38Listwise.READ_TIMEOUT})",
    )
    cli.add_argument(
        "--thinking-budget", type=positive_int, default=Qwen38Listwise.THINKING_BUDGET,
        help=f"Reasoning token budget (default: {Qwen38Listwise.THINKING_BUDGET})",
    )
    cli.add_argument(
        "--max-tokens", type=positive_int, default=Qwen38Listwise.MAX_OUTPUT_TOKENS,
        help=f"Final answer token budget (default: {Qwen38Listwise.MAX_OUTPUT_TOKENS})",
    )
    cli.add_argument(
        "--max-ocr-chars", type=positive_int, help="OCR characters per page (default: full text)",
    )
    cli.add_argument(
        "--max-image-side", type=positive_int, help="Longest page image side in pixels (default: original size)",
    )
    args = cli.parse_args(argv)
    if args.topk > Qwen38Listwise.MAX_CANDIDATES:
        cli.error(f"Local Listwise accepts at most {Qwen38Listwise.MAX_CANDIDATES} candidates per query")

    @contextmanager
    def open_ranker(output_dir):
        with Qwen38Listwise.connect(
            args.base_url, args.read_timeout, thinking_budget=args.thinking_budget,
            max_tokens=args.max_tokens, max_ocr_chars=args.max_ocr_chars,
            max_image_side=args.max_image_side,
        ) as model:
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = output_dir / "checkpoint.sqlite3"
            with closing(sqlite3.connect(checkpoint_path, check_same_thread=False)) as connection:
                checkpoint = Checkpoint(connection)
                yield partial(model.rank_pages, checkpoint=checkpoint)
                print(f"Listwise checkpoint reused {checkpoint.reused} completed queries", flush=True)

    try:
        rerank(args, open_ranker, method="listwise", model_name=Qwen38Listwise.NAME, workers=args.workers)
    except (FileNotFoundError, ValueError, sqlite3.Error) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
