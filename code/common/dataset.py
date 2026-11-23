import argparse
import json
from collections import Counter
from pathlib import Path

from common.docscope import SPLITS, DocScope
from common.mmdocir import MMDocIR
from common.utils import ROOT, group_candidates, open_rgb, positive_int
from common.vidore_v3 import LANGUAGES, SUBSETS, ViDoReV3


DATASETS = {"mmdocir": MMDocIR, "docscope": DocScope, "vidore_v3": ViDoReV3}


def dataset_options(cli):
    cli.add_argument("--dataset-name", choices=DATASETS, default="mmdocir")
    cli.add_argument("--dataset", type=Path, help="Override the dataset directory")
    cli.add_argument("--split", choices=SPLITS, help="DocScope split (default: test)")
    cli.add_argument("--subset", choices=SUBSETS, help="ViDoRe V3 corpus (default: all available)")
    cli.add_argument("--language", choices=LANGUAGES, help="ViDoRe V3 query language (default: native)")
    cli.add_argument("--limit", type=positive_int, help="Only process the first N queries")


def open_dataset(name, directory=None, *, split=None, subset=None, language=None):
    root = Path(directory) if directory is not None else ROOT / "data" / name
    if split is not None and name != "docscope":
        raise ValueError("--split is only supported for DocScope")
    if (subset is not None or language is not None) and name != "vidore_v3":
        raise ValueError("--subset and --language are only supported for ViDoRe V3")
    if name == "docscope":
        return DocScope(root, split=split or "test")
    if name == "vidore_v3":
        return ViDoReV3(root, subset=subset or "all", language=language or "native")
    return DATASETS[name](root)


def dataset_from_args(args):
    return open_dataset(
        args.dataset_name, args.dataset, split=getattr(args, "split", None),
        subset=getattr(args, "subset", None), language=getattr(args, "language", None),
    )


def main(argv=None):
    cli = argparse.ArgumentParser(description="Inspect local dataset annotations and read a sample page")
    dataset_options(cli)
    cli.add_argument("--check-layouts", action="store_true", help="Also load and validate evidence coordinates")
    args = cli.parse_args(argv)
    dataset = dataset_from_args(args)
    queries, _ = dataset.load(args.limit, include_layouts=args.check_layouts)
    page_count = sum(len(dataset.candidate_ids(group[0])) for group in group_candidates(queries))
    sample_id = queries[0]["gt_pages"][0]
    (page_uid, binary), = dataset.images([sample_id])
    with open_rgb(binary) as page:
        sample = {"page_uid": page_uid, "width": page.width, "height": page.height}
    summary = {
        "dataset": args.dataset_name, "queries": len(queries), "candidate_pages": page_count,
        "queries_by_group": dict(sorted(Counter(query["group"] for query in queries).items())),
        "sample_image": sample,
    }
    if args.check_layouts:
        summary["queries_with_boxes"] = sum(bool(query["gt_boxes"]) for query in queries)
        summary["query_page_pairs_with_boxes"] = sum(len(query["gt_boxes"]) for query in queries)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
