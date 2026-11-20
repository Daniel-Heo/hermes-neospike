"""numpy is required. Without it, importing the provider raises ImportError with the pip command."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD_TIMEOUT_SECONDS = 120

# numpy is blocked before any neospike import; the provider import must fail loudly, not fall back.
_CHILD = textwrap.dedent("""
    import sys
    sys.modules["numpy"] = None
    import conftest  # repo root (cwd): the agent.memory_provider stub when Hermes is not on the path
    try:
        import neospike.provider
    except ImportError as error:
        print("ImportError:", error)
    else:
        print("imported")
""")


def test_provider_import_fails_without_numpy(tmp_path):
    completed = subprocess.run([sys.executable, "-c", _CHILD], cwd=REPO_ROOT, capture_output=True, text=True,
                               encoding="utf-8", timeout=CHILD_TIMEOUT_SECONDS,
                               env={**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONIOENCODING": "utf-8"})
    assert completed.returncode == 0, completed.stderr[-3000:]
    last = completed.stdout.strip().splitlines()[-1]
    assert last.startswith("ImportError:") and "Environment without numpy:" in last and "install.py" in last and "needs numpy" in last, last
