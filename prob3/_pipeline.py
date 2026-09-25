"""
CSCI E-89 Assignment 03, Problem 3 -- helper that makes each step runnable on its own
Author: Levente Papp

The step scripts 01-08 are written like notebook cells: they share one
namespace (e.g. 06_train.py uses `model` from 05_model.py). When a step is
run directly, e.g. `python prob3/06_train.py`, run_prerequisites() first
executes all earlier steps in that script's namespace. When the steps are
already running in a shared namespace (run_all.py, or the notebook, where the
bootstrap block is removed) it does nothing.
"""

from pathlib import Path


def step_scripts(folder):
    """Return the step scripts 01_*.py ... 08_*.py in execution order."""
    return sorted(Path(folder).glob("0[1-8]_*.py"))


def run_step(step, namespace):
    """Execute one step script inside `namespace`, printing a banner first."""
    print(f"\n{'=' * 20} {step.name} {'=' * 20}")
    namespace["__file__"] = str(step)
    exec(compile(step.read_text(), str(step), "exec"), namespace)


def run_prerequisites(this_file, namespace):
    """Run every step that comes before `this_file`, unless already done."""
    if namespace.get("_PIPELINE_ACTIVE"):
        return  # earlier steps already ran in this namespace
    namespace["_PIPELINE_ACTIVE"] = True
    this = Path(this_file).resolve()
    for step in step_scripts(this.parent):
        if step.name >= this.name:
            break
        run_step(step, namespace)
    namespace["__file__"] = str(this)
    print(f"\n{'=' * 20} {this.name} {'=' * 20}")
