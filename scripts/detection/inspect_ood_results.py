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


def _glp_layer_matrix(glp: str) -> dict[str, list[float]]:
    """{ood: [auroc per layer 0..15]} for a GLP (empty entries skipped)."""
    out: dict[str, list[float]] = {}
    for o in OOD:
        p = Path(f"results/ood/recon-{glp}-{o}/results.json")
        if not p.exists():
            continue
        per = json.loads(p.read_text()).get("per_layer", {})
        vals = []
        ok = True
        for li in range(16):
            m = per.get(f"layer_{li:02d}") or per.get(f"layer_{li}")
            if m is None:
                ok = False
                break
            vals.append((m.get("recon_error", m))["auroc"])
        if ok:
            out[o] = vals
    return out


def _fixed_layer_report(title: str, mat: dict[str, list[float]]) -> None:
    """Pick the single layer with the best MEAN AUROC across eval-sets; show that row.

    Contrasts with per-cell best-layer (which cherry-picks a different layer per
    eval-set and overstates a method's robustness).
    """
    if not mat:
        return
    oods = list(mat)
    per_layer_mean = [
        sum(mat[o][li] for o in oods) / len(oods) for li in range(16)
    ]
    best_layer = max(range(16), key=lambda li: per_layer_mean[li])
    # also per-cell best (the optimistic view) for comparison
    print(f"\n=== {title}: best FIXED layer across tasks = L{best_layer} "
          f"(mean AUROC {per_layer_mean[best_layer]:.3f}) ===")
    print("  " + f"{'eval-set':<18}{'fixed-L' + str(best_layer):<10}{'per-cell best':<14}")
    for o in oods:
        fixed = mat[o][best_layer]
        cell_best = max(mat[o])
        cell_li = mat[o].index(cell_best)
        print(f"  {o:<18}{fixed:<10.3f}{cell_best:.3f} (L{cell_li})")


def show_fixed_layer() -> None:
    print("\n################ FIXED-LAYER generalization (one layer for all tasks) #####")
    for glp in ("newglp", "origglp"):
        _fixed_layer_report(f"GLP {glp}", _glp_layer_matrix(glp))

    p = Path("results/ood/baselines/results.json")
    if not p.exists():
        return
    regimes = json.loads(p.read_text())["regimes"]
    for method, rows in regimes.items():
        for regime, row in rows.items():
            mat = {
                o: [row[o]["per_layer"][f"layer_{li:02d}"]["auroc"] for li in range(16)]
                for o in OOD
                if o in row and "per_layer" in row[o]
            }
            _fixed_layer_report(f"{method} / {regime}", mat)


if __name__ == "__main__":
    for g in ("newglp", "origglp"):
        show(g)
    show_baselines()
    show_fixed_layer()
