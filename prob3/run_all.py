"""
CSCI E-89 Assignment 03, Problem 3 -- run steps 01-08 in order
Author: Levente Papp

Usage (from the repo root):  python prob3/run_all.py
"""

from pathlib import Path

from _pipeline import run_step, step_scripts

namespace = {"__name__": "__main__", "_PIPELINE_ACTIVE": True}
for step in step_scripts(Path(__file__).resolve().parent):
    run_step(step, namespace)
