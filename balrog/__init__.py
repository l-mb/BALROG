import io
import sys

# Suppress gym deprecation stderr message during import
# Migration to gymnasium deferred - dependencies still require old gym
_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    import gym  # noqa: F401
finally:
    sys.stderr = _stderr
del _stderr
