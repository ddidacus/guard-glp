"""Per-layer AUROC across depth for the GLPs and the supervised baselines.

Reads existing results.json files (no re-run). Prints per-layer AUROC tables to stdout
and writes line-plot figures (AUROC vs layer 0-15) to results/ood/figures/.

  GLP recon      : per_layer already saved by evaluate_classifier.aggregate.
  probe/diffmean : per_layer saved by eval_ood_baselines.aggregate (re-run that
                   aggregate once after the per-layer change — CPU only, cached acts).

    python scripts/detection/plot_ood_layers.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OOD = ["advbench", "harmbench", "wjb_vanilla", "wjb_adversarial", "toxicchat"]
LAYERS = list(range(16))
FIG_DIR = Path("results/ood/figures")

# fixed categorical order (dataviz: assign in order, never cycle). Distinct, CVD-aware.
_COLORS = ["#3b6fd6", "#e8710a", "#12a594", "#d1478c", "#8a6ee0", "#6b7280", "#b59a00"]


def _glp_layer_auroc(glp: str, ood: str) -> list[float] | None:
    p = Path(f"results/ood/recon-{glp}-{ood}/results.json")
    if not p.exists():
        return None
    per = json.loads(p.read_text()).get("per_layer", {})
    out: list[float] = []
    for li in LAYERS:
        m = per.get(f"layer_{li:02d}") or per.get(f"layer_{li}")
        if m is None:
            return None
        out.append((m.get("recon_error", m))["auroc"])
    return out


def _baseline_layer_auroc(method: str, regime: str, ood: str) -> list[float] | None:
    p = Path("results/ood/baselines/results.json")
    if not p.exists():
        return None
    reg = json.loads(p.read_text())["regimes"].get(method, {}).get(regime, {})
    cell = reg.get(ood, {})
    per = cell.get("per_layer")
    if not per:
        return None
    return [per[f"layer_{li:02d}"]["auroc"] for li in LAYERS]


def _print_table(title: str, rows: dict[str, list[float]]) -> None:
    print(f"\n=== {title} — AUROC by layer ===")
    print("  " + f"{'series':<22}" + "".join(f"L{li:<4}" for li in LAYERS))
    for name, vals in rows.items():
        print("  " + f"{name:<22}" + "".join(f"{v:<5.2f}" for v in vals))


def _plot(title: str, rows: dict[str, list[float]], fname: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (name, vals), color in zip(rows.items(), _COLORS, strict=False):
        ax.plot(LAYERS, vals, "-", color=color, lw=2, marker="o", ms=4, label=name)
    ax.axhline(0.5, color="#9ca3af", ls="--", lw=1)  # chance
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.set_xticks(LAYERS)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", color="#eee", lw=0.8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  saved {FIG_DIR / fname}")


def main() -> None:
    # GLP: new vs orig, one figure/table per OOD set
    for ood in OOD:
        rows = {}
        for glp in ("newglp", "origglp"):
            v = _glp_layer_auroc(glp, ood)
            if v:
                rows[glp] = v
        if rows:
            _print_table(f"GLP recon — {ood}", rows)
            _plot(f"GLP recon AUROC by layer — {ood}", rows, f"glp_{ood}.png")

    # Baselines: for each method+eval-set, one line per training regime
    for method in ("probe", "diffmean"):
        for ood in OOD:
            rows = {}
            bp = Path("results/ood/baselines/results.json")
            if bp.exists():
                for regime in json.loads(bp.read_text())["regimes"].get(method, {}):
                    v = _baseline_layer_auroc(method, regime, ood)
                    if v:
                        rows[regime] = v
            if rows:
                _print_table(f"{method} — eval:{ood} (by regime)", rows)
                _plot(
                    f"{method} AUROC by layer — eval:{ood}",
                    rows,
                    f"{method}_{ood}.png",
                )


if __name__ == "__main__":
    main()
