# tests/test_mypy.py
import subprocess
import shutil
import pytest

def test_mypy_clean():
    if shutil.which("mypy") is None:
        pytest.skip("mypy не установлен")
    result = subprocess.run(
        ["mypy", "core"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"Mypy errors:\n{result.stdout}\n{result.stderr}"