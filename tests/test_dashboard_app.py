"""Smoke and integration tests for dashboard Streamlit frontend."""

import ast
from pathlib import Path
import py_compile
import runpy
from unittest.mock import MagicMock, patch

import numpy
import pandas
import pytest


APP_PATH = Path("dashboard/app.py")


def test_app_compilation():
    """Validates that dashboard/app.py compiles without syntax errors."""
    assert APP_PATH.exists()
    compiled_file = py_compile.compile(str(APP_PATH), doraise=True)
    assert compiled_file is not None


def test_app_static_structure():
    """Statically verifies required operational controls and panels in app.py."""
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Confirm PyTorch / ultralytics are NOT imported
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module)

    assert "torch" not in imported_names
    assert "ultralytics" not in imported_names

    # Check for core panels in source
    required_phrases = [
        "AI Drone Swarm Command Center",
        "SIMULATED DATA",
        "Network Map",
        "Drone Status",
        "Threat & Quarantine",
        "Self-Healing",
        "Disaster Detection",
        "Post-Quantum Security",
        "Telemetry Charts",
        "Audit Event Log",
        "is_post_quantum",
        "CLASSICAL FALLBACK",
        "hazards is None",
        "Not implemented yet",
        "Mobility simulator not implemented yet",
    ]
    for phrase in required_phrases:
        assert phrase.lower() in source.lower(), f"Missing expected dashboard feature or label: '{phrase}'"


class _SessionState(dict):
    """Dictionary supporting attribute access for st.session_state mocking."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class _DummyContext:
    """Mock context manager for st.sidebar, st.container, st.expander, st.tabs."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_app_execution_smoke():
    """Executes app.py top-to-bottom under a mocked Streamlit harness."""
    mock_st = MagicMock()
    mock_st.sidebar = _DummyContext()
    mock_st.columns.side_effect = lambda spec, *args, **kwargs: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    mock_st.tabs.side_effect = lambda names, *args, **kwargs: [
        _DummyContext() for _ in names
    ]
    mock_st.container.return_value = _DummyContext()
    mock_st.expander.return_value = _DummyContext()
    mock_st.session_state = _SessionState()
    mock_st.radio.return_value = "Interactive SVG (Fast)"
    mock_st.button.return_value = False
    mock_st.selectbox.return_value = "D4"

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        runpy.run_path(str(APP_PATH), run_name="__main__")

    # Assert 8 tabs were created
    assert mock_st.tabs.call_count >= 1
    tab_names = mock_st.tabs.call_args[0][0]
    assert len(tab_names) == 8

    # Assert DataFrames were rendered (drone table, event log, etc.)
    assert mock_st.dataframe.call_count >= 1

    # Assert metrics rendered
    assert mock_st.metric.call_count >= 1


def test_app_execution_matplotlib_mode():
    """Executes app.py with Matplotlib rendering mode selected."""
    mock_st = MagicMock()
    mock_st.sidebar = _DummyContext()
    mock_st.columns.side_effect = lambda spec, *args, **kwargs: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    mock_st.tabs.side_effect = lambda names, *args, **kwargs: [
        _DummyContext() for _ in names
    ]
    mock_st.container.return_value = _DummyContext()
    mock_st.expander.return_value = _DummyContext()
    mock_st.session_state = _SessionState()
    mock_st.radio.return_value = "Matplotlib"
    mock_st.button.return_value = False
    mock_st.selectbox.return_value = "D4"

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        runpy.run_path(str(APP_PATH), run_name="__main__")

    # Assert st.pyplot was invoked
    assert mock_st.pyplot.call_count >= 1
