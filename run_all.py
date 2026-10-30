"""Runs the pipeline in dependency order:
    01 benchmark -> 04 statistics -> 06 critical-constant confound -> 07 in-sample fit -> 08 paired statistics
    -> 09 in-sample evaluation -> 10 PR78, volume shift, input error -> 11 physical diagnostics
    -> 02 tables -> 03 figures -> 05 checks

Usage:  python run_all.py            full run (about 30 min; the in-sample fit takes 20-25 min)
        python run_all.py --quick    smoke test (about 3 min): in-sample fit on a subset, no tables/figures/checks
        python run_all.py --from 02  start at a given script (earlier outputs must exist)
Every script writes into data/, tables/ or figures/; a clean run reproduces the committed files (fixed seed).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
ORDER = ["01_run_benchmark.py", "04_stats.py", "06_twufit_source_constants.py", "07_insample_fit.py",
         "08_paired_statistics.py", "09_insample_evaluation.py", "10_pr78_shift_input_error.py",
         "11_physical_diagnostics.py", "02_make_tables.py", "03_make_figures.py", "05_check_manuscript_numbers.py"]

quick = "--quick" in sys.argv
start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "01"
first = next(i for i, x in enumerate(ORDER) if x.startswith(start))
scripts = ORDER[first:]
if quick:
    scripts = [s for s in scripts if s[:2] in ("01", "04", "06", "07")]
t_all = time.time()
for script in scripts:
    args = [sys.executable, os.path.join(CODE, script)]
    if quick and script.startswith("07"):
        args.append("--quick")
    print(f"\n==== {script}", flush=True)
    t0 = time.time()
    r = subprocess.run(args, cwd=CODE)
    print(f"==== {script} finished in {time.time() - t0:.0f} s (exit {r.returncode})", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed")
print(f"\nall done in {time.time() - t_all:.0f} s")
