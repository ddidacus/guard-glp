"""Diagnostic: print per-layer good/bad recon-error means + AUROC for each OOD task.

Reveals whether the score direction is inverted (bad_mean < good_mean => OOD reconstructs
BETTER than ID => below-chance AUROC), and at which layers. CPU-only, reads results.json.

    python scripts/detection/inspect_ood_results.py
"""

import json
from pathlib import Path

OOD = ["advbench", "harmbench", "wjb_vanilla", "wjb_adversarial", "toxicchat"]


def show(glp: str) -> None:
    print(f"\n================ {glp} ================")
    for o in OOD:
        p = Path(f"results/ood/recon-{glp}-{o}/results.json")
        if not p.exists():
            print(f"{o:16s}  (no results.json)")
            continue
        r = json.loads(p.read_text())
        per = r.get("per_layer", {})
        # find the row for each layer: metric nested under 'recon_error'
        print(f"\n{o}:")
        print(f"  {'layer':<6}{'good_mean':<12}{'bad_mean':<12}{'AUROC':<8}{'AUPRC':<8}")
        for lname, m in per.items():
            rm = m.get("recon_error", m)
            gm = rm.get("good_mean", float("nan"))
            bm = rm.get("bad_mean", float("nan"))
            print(
                f"  {lname.replace('layer_', ''):<6}{gm:<12.3f}{bm:<12.3f}"
                f"{rm.get('auroc', float('nan')):<8.3f}{rm.get('auprc', float('nan')):<8.3f}"
            )


def show_baselines() -> None:
    p = Path("results/ood/baselines/results.json")
    if not p.exists():
        print("\n(no baseline results.json)")
        return
    regimes = json.loads(p.read_text())["regimes"]
    for metric in ("auprc", "auroc"):
        print(f"\n================ baselines — {metric.upper()} (best layer) ================")
        for method, rows in regimes.items():
            print(f"\n[{method}]")
            print("  " + f"{'regime':<20}" + "".join(f"{o[:12]:<14}" for o in OOD))
            for regime, row in rows.items():
                cells = "".join(f"{row[o][metric]:<14.3f}" for o in OOD)
                print(f"  {regime:<20}{cells}")


if __name__ == "__main__":
    for g in ("newglp", "origglp"):
        show(g)
    show_baselines()
