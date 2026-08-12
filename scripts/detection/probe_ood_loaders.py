"""Quick CPU-only check that every OOD loader resolves (right HF id / columns).

Run before the full benchmark to catch dataset schema/access issues cheaply:
    python scripts/detection/probe_ood_loaders.py

Prints one line per OOD set: OK with train/cal/test sizes, or FAIL with the error
(the loaders fail loud with the real schema on a column mismatch). Also probes the ID
pool.
"""

from glp.dataset.ood_prompts import OOD_SETS, load_id_pool, load_ood_pool


def main() -> None:
    try:
        p = load_id_pool()
        print(f"{'id_pool':18s} OK  {len(p.train)}/{len(p.cal)}/{len(p.test)}")
    except Exception as e:  # noqa: BLE001 - report any failure, keep going
        print(f"{'id_pool':18s} FAIL {type(e).__name__}: {str(e)[:160]}")

    for n in OOD_SETS:
        try:
            p = load_ood_pool(n)
            print(f"{n:18s} OK  {len(p.train)}/{len(p.cal)}/{len(p.test)}")
        except Exception as e:  # noqa: BLE001 - report any failure, keep going
            print(f"{n:18s} FAIL {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
