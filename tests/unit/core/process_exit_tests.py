#!/usr/bin/env python3
"""Real process exit policy; no native driver or installed user state."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="studiocast-exit-") as directory:
    env = dict(os.environ)
    for key, name in (("HOME", "home"), ("XDG_CONFIG_HOME", "config"),
                      ("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "cache"),
                      ("XDG_STATE_HOME", "state"), ("XDG_RUNTIME_DIR", "runtime")):
        path = Path(directory) / name
        path.mkdir(mode=0o700)
        env[key] = str(path)
    for mode, status, expected in (("normal", "zero", 0),
                                   ("normal", "failure", 7),
                                   ("retain", "zero", 1),
                                   ("retain", "failure", 1),
                                   ("normal", "skip", 77),
                                   ("retain", "skip", 1)):
        result = subprocess.run([binary, mode, status], env=env, capture_output=True,
                                text=True, timeout=5, check=False)
        assert result.returncode == expected, (mode, status, result)
        if mode == "retain":
            assert result.stdout == "owner_retained\nlocal_cleanup\n", result
            assert "unsafe native resources retained" in result.stderr, result
        else:
            assert result.stdout == "local_cleanup\natexit\n", result
            assert not result.stderr, result
print("process exit policy: six subprocess cases passed")
