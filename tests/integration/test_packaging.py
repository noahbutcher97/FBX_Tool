"""Integration tests for distributable package artifacts."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _copy_packaging_source(repo_root: Path, source_dir: Path) -> None:
    """Copy the packaging inputs into an isolated build source tree."""
    source_dir.mkdir()

    for filename in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "MANIFEST.in",
        "setup.py",
    ):
        source_file = repo_root / filename
        if source_file.exists():
            shutil.copy2(source_file, source_dir / filename)

    shutil.copytree(
        repo_root / "fbx_tool",
        source_dir / "fbx_tool",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


@pytest.mark.integration
def test_built_wheel_includes_runtime_subpackages(tmp_path):
    """Built wheels should include every importable runtime subpackage."""
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = tmp_path / "source"
    wheel_dir = tmp_path / "wheelhouse"
    _copy_packaging_source(repo_root, source_dir)
    wheel_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(source_dir),
            "--no-deps",
            "-w",
            str(wheel_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    wheels = list(wheel_dir.glob("fbx_tool-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        packaged_files = set(wheel.namelist())

    expected_runtime_modules = {
        "fbx_tool/__init__.py",
        "fbx_tool/__main__.py",
        "fbx_tool/analysis/__init__.py",
        "fbx_tool/analysis/utils.py",
        "fbx_tool/gui/__init__.py",
        "fbx_tool/gui/main_window.py",
        "fbx_tool/visualization/__init__.py",
        "fbx_tool/visualization/opengl_viewer.py",
    }

    assert expected_runtime_modules <= packaged_files
