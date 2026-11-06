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


def test_run_self_healing_demo_interactive_mode() -> None:
    """Verifies that run_self_healing_demo.py --interactive executes cleanly across user pauses."""
    proc = subprocess.Popen(
        [sys.executable, "run_self_healing_demo.py", "--interactive"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate(input="\n" * 10, timeout=15)
    assert proc.returncode == 0
    assert "Demonstration completed successfully!" in stdout
    assert "Error: Source node 'D7' is in the excluded set." not in stderr
    assert "Source node 'D7' is in the excluded set" not in stdout


