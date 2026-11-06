"""Tests for the interactive self-healing demonstration runner."""

import subprocess
import sys


def test_run_self_healing_demo_executes_cleanly() -> None:
    """Verifies that run_self_healing_demo.py executes with 0 exit code."""
    res = subprocess.run(
        [sys.executable, "run_self_healing_demo.py", "--fast"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "AI DRONE SWARM: SELF-HEALING CONTROLLER DEMONSTRATION" in res.stdout
    assert "PHASE 1" in res.stdout
    assert "PHASE 2" in res.stdout
    assert "PHASE 3" in res.stdout
    assert "PHASE 4" in res.stdout
    assert "PHASE 5" in res.stdout
    assert "PHASE 6" in res.stdout
    assert "Demonstration completed successfully!" in res.stdout

