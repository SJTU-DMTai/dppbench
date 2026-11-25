"""Print paired std_test AUC and original exp1 cost fields, one row per run."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

TASKS = ("polish_bankruptcy", "home_credit", "fraud_detection")
BASELINES = ("Random", "SAGA", "AlphaClean", "Learn2Clean", "AutoPrep", "DAGPipe", "DiffPrep")


def read_result(path, task, method):
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"^\[FAILED\]", raw, re.M):
        return None
    decoder = json.JSONDecoder()
    for start in reversed(list(re.finditer(r'^\{\s*"results_csv"', raw, re.M))):
        try:
            summary, _ = decoder.raw_decode(raw[start.start():])
        except ValueError:
            continue
        for row in summary["results"]:
            if (row["task"] == task and row["baseline"] == method and row["model"] == "LightGBM"
                    and not row.get("error") and (row.get("std_test") or {}).get("auc") is not None):
                return row
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("clean","mcar","mar","savage"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    writer = csv.writer(sys.stdout)
    groups = vars(args)
    metrics = ("auc","n_rows","search_or_training_s","final_training_s","final_inference_s")
    writer.writerow(["task","baseline","seed",*(f"{g}_{m}" for g in groups for m in metrics),
                     "savage_minus_clean","savage_minus_mcar","savage_minus_mar","missing"])
    counts = dict.fromkeys(groups,0)
    for task in TASKS:
        for method in BASELINES:
            for seed in (42,43,44):
                line = [task,method,seed]
                aucs, sizes = {}, set()
                for name, root in groups.items():
                    row = read_result(root/f"seed{seed}/{task}_{method}_seed{seed}.log",task,method)
                    if row is None:
                        line.extend([""]*len(metrics))
                        continue
                    counts[name] += 1
                    test, cost = row["std_test"], row["time_cost_seconds"]
                    aucs[name] = test["auc"]
                    sizes.add(test["n_rows"])
                    search = cost.get("baseline_search_inference")
                    if search is None:
                        search = cost.get("baseline_training")
                    line.extend([test["auc"],test["n_rows"],search,cost.get("final_model_training"),
                                 cost.get("final_model_inference")])
                if len(sizes)>1:
                    raise ValueError(f"Held-out counts differ for {task}/{method}/{seed}: {sizes}")
                for name in ("clean","mcar","mar"):
                    line.append(aucs["savage"]-aucs[name] if name in aucs and "savage" in aucs else "")
                writer.writerow([*line,",".join(g for g in groups if g not in aucs)])
    print("Available valid summaries out of 63:", counts, file=sys.stderr)


if __name__=="__main__":
    main()
