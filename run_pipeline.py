"""One-command pipeline runner.

Usage
-----
    python run_pipeline.py                 # everything
    python run_pipeline.py --stage features
    python run_pipeline.py --stage train
    python run_pipeline.py --stage cv
    python run_pipeline.py --stage evaluate
    python run_pipeline.py --force         # ignore cached artifacts
"""

import argparse
import time

from src import evaluation, train_models

STAGES = ["features", "train", "cv", "evaluate"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=["all"] + STAGES, default="all")
    parser.add_argument("--force", action="store_true",
                        help="rebuild artifacts even if they exist")
    args = parser.parse_args()

    stage = args.stage
    t_start = time.perf_counter()

    if stage in ("all", "features"):
        train_models.build_features(force=args.force)
    if stage in ("all", "train"):
        train_models.train_models(force=args.force)
    if stage in ("all", "cv"):
        train_models.cross_validate(force=args.force)
    if stage in ("all", "evaluate"):
        evaluation.run_all()

    mins = (time.perf_counter() - t_start) / 60
    print(f"\n[run_pipeline] finished stage '{stage}' in {mins:.1f} min")


if __name__ == "__main__":
    main()
