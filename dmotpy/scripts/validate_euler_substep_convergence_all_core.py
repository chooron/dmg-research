from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.euler_convergence_all_core_utils import (
    ALL_CORE_TARGET_MODELS,
    EXCLUDED_MODELS,
    run_euler_convergence_validation_all_core,
)


def main() -> int:
    print("=== Euler Substep First-Order Convergence — All Core Models ===")
    print(f"Target models ({len(ALL_CORE_TARGET_MODELS)}): {', '.join(ALL_CORE_TARGET_MODELS)}")
    print(f"Excluded models ({len(EXCLUDED_MODELS)}): {', '.join(sorted(EXCLUDED_MODELS))}")
    print()

    artifacts = run_euler_convergence_validation_all_core(write_outputs=True)

    summary_rows = artifacts["summary_rows"]
    pass_rows = [r for r in summary_rows if r["state_convergence_pass"]]
    fail_rows = [r for r in summary_rows if not r["state_convergence_pass"]]

    print(f"Passed: {len(pass_rows)} / {len(summary_rows)}")
    print()

    if fail_rows:
        print("FAILURES:")
        for row in sorted(fail_rows, key=lambda r: r["model"]):
            print(
                f"  {row['model']}: classification={row['classification']}  "
                f"median_p_state={row['median_p_state']:.3f}  "
                f"monotone={row['state_error_monotone']}  "
                f"notes={row['notes']}"
            )
        print()

    print("Per-model results:")
    for row in sorted(summary_rows, key=lambda r: r["model"]):
        mp = row["median_p_state"]
        mp_str = f"{mp:.3f}" if isinstance(mp, float) and mp == mp else "n/a"
        status = "PASS" if row["state_convergence_pass"] else "FAIL"
        print(
            f"  [{status}] {row['model']:20s}  median_p_state={mp_str}  "
            f"monotone={str(row['state_error_monotone']):5s}  "
            f"class={row['classification']}"
        )

    from tests.euler_convergence_all_core_utils import (
        ALL_CORE_ERRORS_CSV_PATH,
        ALL_CORE_ORDERS_CSV_PATH,
        ALL_CORE_SUMMARY_CSV_PATH,
        ALL_CORE_REPORT_MD_PATH,
    )
    print()
    print("Wrote outputs:")
    print(f"  {ALL_CORE_ERRORS_CSV_PATH}")
    print(f"  {ALL_CORE_ORDERS_CSV_PATH}")
    print(f"  {ALL_CORE_SUMMARY_CSV_PATH}")
    print(f"  {ALL_CORE_REPORT_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
