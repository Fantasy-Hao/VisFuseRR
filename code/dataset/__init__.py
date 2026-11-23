import argparse
from pathlib import Path

from utils import ROOT, positive_int
from .docscope import DocScope, SPLITS
from .mmdocir import MMDocIR
from .vidore_v3 import ViDoReV3, SUBSETS


DATASETS = {"mmdocir": MMDocIR, "docscope": DocScope, "vidore_v3": ViDoReV3}


def parser(description):
    cli = argparse.ArgumentParser(description=description)
    cli.add_argument("--dataset-name", choices=DATASETS, default="mmdocir")
    cli.add_argument("--dataset", type=Path, help="Override the local dataset directory")
    cli.add_argument("--split", choices=SPLITS, help="DocScope split (default: test)")
    cli.add_argument("--subset", choices=SUBSETS, help="ViDoRe V3 corpus (default: all five)")
    cli.add_argument("--limit", type=positive_int, help="Use the first N queries")
    cli.add_argument("--output", type=Path, default=ROOT / "output")
    return cli


def dataset_from_args(args):
    options = {}
    if args.split:
        if args.dataset_name != "docscope":
            raise ValueError("--split applies to DocScope")
        options["split"] = args.split
    if args.subset:
        if args.dataset_name != "vidore_v3":
            raise ValueError("--subset applies to ViDoRe V3")
        options["subset"] = args.subset
    root = args.dataset or ROOT / "data" / args.dataset_name
    return DATASETS[args.dataset_name](root, **options)
