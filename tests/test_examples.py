"""The scripts in ``examples/`` must run.

They are the library's quickstart, and they are the documentation most likely to rot: nothing
imports them, so an API change breaks them silently. (The example this replaced had been broken
since ``get_samples()`` began returning a dict, and nothing noticed.)

Each test runs a script end to end in a subprocess -- the way a reader runs it -- and checks both
that it exits cleanly and that the line where it reports its result is present. The examples
therefore stay free of test hooks: no quick mode, no importable entry point, no environment flags.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_example(name: str, timeout: int = 600) -> str:
    """Run ``examples/<name>`` and return its stdout, failing on a non-zero exit."""
    script = EXAMPLES / name
    assert script.is_file(), f"missing example: {script}"
    proc = subprocess.run([sys.executable, str(script)], cwd=str(EXAMPLES.parent),
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        pytest.fail(f"{name} exited {proc.returncode}\n"
                    f"--- stdout ---\n{proc.stdout[-4000:]}\n"
                    f"--- stderr ---\n{proc.stderr[-4000:]}")
    return proc.stdout


def test_quickstart():
    out = run_example("01_quickstart.py")
    assert "Sample summary" in out
    assert "Recovered vs truth" in out


def test_model_by_hand():
    out = run_example("02_model_by_hand.py")
    # The example's own claim: the hand-built model and the DSL one are the same density.
    assert "log density at a test point" in out
    assert "Recovered vs truth" in out


def test_factory_and_evidence():
    out = run_example("03_factory_and_evidence.py")
    assert "Recovered vs truth" in out
    # The evidence round must actually reach the factory, which is the point of the example.
    assert "learned_metric" in out


def test_sampler_by_hand():
    out = run_example("04_sampler_by_hand.py")
    assert "Recovered vs truth" in out


def test_mixture():
    out = run_example("05_mixture.py")
    # The example's own claims: the labels are sampled as integers, and they move. A frozen
    # sweep would still print recovered means (from the continuous block alone) and a *perfect*
    # ESS, so the move count is the assertion that can actually fail.
    assert "label agreement with the generating assignment" in out
    assert "label moves per iteration" in out
    moves = float(out.split("label moves per iteration:")[1].split("of")[0])
    assert moves > 0.5, f"the discrete sweep is stuck ({moves} moves/iteration)"
