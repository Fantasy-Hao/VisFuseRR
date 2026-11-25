"""Dataset adapters."""

import argparse
from pathlib import Path

from utils import ROOT, positive_int

from .docscope import DocScope, SPLITS
from .longdocurl import LongDocURL
from .mmdocir import MMDocIR


DATASETS = {"mmdocir": MMDocIR, "docscope": DocScope, "longdocurl": LongDocURL}


def parser(description):
    cli = argparse.ArgumentParser(description=description)
    cli.add_argument("--dataset-name", choices=DATASETS, default="mmdocir")
    cli.add_argument("--dataset", type=Path, help="Override the local dataset directory")
    cli.add_argument("--split", choices=SPLITS, help="DocScope split (default: test)")
    cli.add_argument("--limit", type=positive_int, help="Use the first N queries")
    cli.add_argument("--output", type=Path, default=ROOT / "output", help="Inference output directory")
    return cli


def dataset_from_args(args):
    root = args.dataset or ROOT / "data" / args.dataset_name
    if args.split is not None:
        if args.dataset_name != "docscope":
            raise ValueError("--split applies to DocScope")
        return DocScope(root, split=args.split)
    return DATASETS[args.dataset_name](root)
