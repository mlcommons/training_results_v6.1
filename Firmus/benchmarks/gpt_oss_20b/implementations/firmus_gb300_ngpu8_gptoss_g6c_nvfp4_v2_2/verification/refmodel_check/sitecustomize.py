# Classification: Internal
# Loaded by Python at start-up when this directory is on PYTHONPATH (launch_refmodel_check.sh).
# Runs any sitecustomize the image already has, then, with REFCHECK=1, installs the check.
import importlib.machinery
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.machinery.PathFinder.find_spec(
    "sitecustomize", [p for p in sys.path if os.path.abspath(p or ".") != _here]
)
if _spec is not None and _spec.origin and os.path.dirname(os.path.abspath(_spec.origin)) != _here:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

if os.environ.get("REFCHECK") == "1":
    import refmodel_check

    refmodel_check.install()
