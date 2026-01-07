import os

import csv
import time
import numpy as np
import re

from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utils.options import parse_args
from utils.util import get_split_loader, set_seed

from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler


def _parse_fold_order(fold_arg: str, n_folds: int = 5) -> list[int]:
    fold_arg = (fold_arg or "").strip()
    if fold_arg == "":
        return list(range(n_folds))

    if fold_arg.isdigit():
        folds = [int(ch) for ch in fold_arg]
    else:
        parts = [p for p in re.split(r"[,\s]+", fold_arg) if p]
        folds = [int(p) for p in parts]

    if not folds:
        return list(range(n_folds))

    if any(f < 0 or f >= n_folds for f in folds):
        raise ValueError(f"--fold must be in [0,{n_folds - 1}] (got {fold_arg!r})")

    unique_folds: list[int] = []
    seen = set()
    for f in folds:
        if f in seen:
            continue
        seen.add(f)
        unique_folds.append(f)
    return unique_folds


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)

    # create results directory
    results_dir = "./results/{dataset}/[{model}]-[{fusion}]-[{alpha}]-[{time}]".format(
        dataset=args.dataset,
        model=args.model,
        fusion=args.fusion,
        alpha=args.alpha,
        time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    fold_order = _parse_fold_order(args.fold, n_folds=5)
    print(f"Running folds in order: {fold_order}")

    header = ["folds"] + [f"fold {f}" for f in fold_order] + ["mean", "std"]
    best_epoch = ["best epoch"]
    best_score = ["best cindex"]

    # start 5-fold CV evaluation.
    for fold in fold_order:
        set_seed(args.seed + fold)
        # build dataset
        dataset = Generic_MIL_Survival_Dataset(
            csv_path="./csv/%s_all_clean.csv" % (args.dataset),
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.data_root_dir,
            shuffle=False,
            seed=args.seed,
            patient_strat=False,
            n_bins=4,
            label_col="survival_months",
        )
        split_dir = os.path.join("./splits", args.which_splits, args.dataset)
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(split_dir, fold)
        )
        train_loader = get_split_loader(
            train_dataset,
            training=True,
            weighted=args.weighted_sample,
            modal=args.modal,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
        )
        val_loader = get_split_loader(
            val_dataset,
            modal=args.modal,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
        )
        print(
            "training: {}, validation: {}".format(len(train_dataset), len(val_dataset))
        )

        # build model, criterion, optimizer, schedular
        if args.model == "cmta":
            from models.cmta.network import CMTA
            from models.cmta.engine import Engine

            print(train_dataset.omic_sizes)
            model_dict = {
                "omic_sizes": train_dataset.omic_sizes,
                "n_classes": 4,
                "fusion": args.fusion,
                "model_size": args.model_size,
            }
            model = CMTA(**model_dict)
            criterion = define_loss(args)
            optimizer = define_optimizer(args, model)
            scheduler = define_scheduler(args, optimizer)
            engine = Engine(args, results_dir, fold)
        else:
            raise NotImplementedError(
                "Model [{}] is not implemented".format(args.model)
            )
        # start training
        score, epoch = engine.learning(
            model, train_loader, val_loader, criterion, optimizer, scheduler
        )
        # save best score and epoch for each fold
        best_epoch.append(epoch)
        best_score.append(score)

    # finish training
    # mean and std
    fold_scores = np.asarray(best_score[1:], dtype=np.float64)
    best_epoch.append("~")
    best_epoch.append("~")
    best_score.append(float(fold_scores.mean()))
    best_score.append(float(fold_scores.std()))

    csv_path = os.path.join(results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_epoch)
        writer.writerow(best_score)


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
