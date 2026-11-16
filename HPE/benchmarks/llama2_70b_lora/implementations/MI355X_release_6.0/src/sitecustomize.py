"""Auto-loaded by CPython at interpreter startup.

Llama 2 launches through primus-cli rather than our own entrypoint, so there is
no convenient place to import the NUMA helper. Putting it in sitecustomize makes
every rank apply it as long as this directory is on PYTHONPATH.
"""
try:
    import _numa_affinity  # noqa: F401
except Exception:
    pass
