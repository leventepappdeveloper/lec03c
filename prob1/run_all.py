"""
CSCI E-89 Assignment 03, Problem 1 -- run all steps from the terminal
Author: Levente Papp

The step scripts are written as notebook cells that share one namespace, so
they can't be run one at a time with `python3 prob1/0X_....py`. This runner
executes them in order in a shared namespace, the same way a notebook does.

Usage (from the repo root):
    python3 prob1/run_all.py            # all steps, 1-8
    python3 prob1/run_all.py 1 2 7 8    # only some steps (7 and 8 read the
                                        # files that step 6 saved)
"""

import sys
from pathlib import Path

PROB1_DIR = Path(__file__).resolve().parent
STEPS = sorted(PROB1_DIR.glob("0[1-8]_*.py"))

wanted = {int(arg) for arg in sys.argv[1:]} or set(range(1, len(STEPS) + 1))
namespace = {"__name__": "__main__"}
for step in STEPS:
    n = int(step.name[:2])
    if n not in wanted:
        continue
    print(f"\n{'=' * 20} Step {n}: {step.name} {'=' * 20}")
    exec(compile(step.read_text(), str(step), "exec"), namespace)
