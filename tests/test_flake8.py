import subprocess
import shutil
import pytest

def test_flake8_clean():
    if shutil.which("flake8") is None:
        pytest.skip("flake8 не установлен")
    result = subprocess.run(
        ["flake8", "core", "tests"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"Flake8 errors:\n{result.stdout}\n{result.stderr}"