"""Firmus auto_kernel modules for MLPerf Training v6.1 llama31_8b, workstream L4 v3 (implementation nemo_l4_nvfp4_v3).

The recipe's pretrain.py imports this package as its first import (firmus/patches/recipe/0002), before torch,
Transformer Engine or Megatron are imported.  First, banner re-hashes the built kernels and the patched upstream files
against the image's build records and raises on any difference.  Then the eight Firmus patch modules are imported in
the order the validated runs loaded them (their zz-ak-l8b-*.pth files, in site.py's sorted order): fp4g, fp8cs, gsa,
lmh, nvfp4p, opswz, projcol, smallk2.  Each module only installs import hooks, and only when its AK_* switch is 1
(config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh); with every switch unset they change nothing.  The modules keep their top-level
names, so their own cross-references (nvfp4p asks sys.modules for ak_l8b_fp4g_patch) and their proof lines are unchanged.

Last, run_start_gate makes the end of the recipe's warm-up fail loudly if a part whose switch is on never armed.
"""
from . import banner  # noqa: F401  (first: the image's bytes are checked before anything else is loaded)

import ak_l8b_fp4g_patch  # noqa: F401,E402
import ak_l8b_fp8cs_patch  # noqa: F401,E402
import ak_l8b_gsa  # noqa: F401,E402
import ak_l8b_lmh_patch  # noqa: F401,E402
import ak_l8b_nvfp4p_patch  # noqa: F401,E402
import ak_l8b_opswz  # noqa: F401,E402
import ak_l8b_projcol  # noqa: F401,E402
import ak_l8b_smallk2_patch  # noqa: F401,E402

from . import run_start_gate  # noqa: F401,E402
