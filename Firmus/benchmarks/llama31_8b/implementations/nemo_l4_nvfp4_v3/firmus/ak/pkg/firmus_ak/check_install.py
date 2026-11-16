"""Install check of the Firmus L4 kernels, run by the Dockerfile after build_all.py and the firmus_ak install, as a script:

    python3 /workspace/llm/firmus/ak/pkg/firmus_ak/check_install.py --prefix /opt/firmus/ak

Standard library only, and it never imports the firmus_ak package (whose banner needs the image's build records, which
are written after this check): the installed package and the eight patch modules are located with importlib's
find_spec, which does not execute them, and read as files.  CUDA-free.  It fails (exit status 1, every failure listed) if
  1. a Firmus .pth file is in site-packages (zz-ak-*.pth, or any .pth naming ak_l8b_ / firmus_ak / auto_kernel): the
     kernels are enabled only by pretrain.py's `import firmus_ak` (gap B-4);
  2. one of the 11 AK_*_DIR variables the Dockerfile sets is unset, resolves outside --prefix, or is not the directory
     EXPECTED_OUTPUTS.json's env_dirs gives it (gap B-5);
  3. a cluster path (the cluster's home, shared or vast mounts) appears in firmus_ak or in one of the eight patch
     modules (gap B-5; the pattern of tools/check_impl_hygiene.py H1);
  4. a file a Firmus module or patch loads is missing or empty: every entry of EXPECTED_OUTPUTS.json's "files" (written
     by build_all.py from build_manifest.json and installed at <prefix>/EXPECTED_OUTPUTS.json; a missing copy is itself
     a failure -- nothing falls back).
"""
import argparse
import importlib.util
import json
import os
import re
import site
import sys

MODULES = ("ak_l8b_fp4g_patch", "ak_l8b_fp8cs_patch", "ak_l8b_gsa", "ak_l8b_lmh_patch", "ak_l8b_nvfp4p_patch",
           "ak_l8b_opswz", "ak_l8b_projcol", "ak_l8b_smallk2_patch")
# the 11 variables of the Dockerfile ENV (section 5); EXPECTED_OUTPUTS.json's env_dirs must name exactly these
ENV_VARS = ("AK_X12_KERNEL_DIR", "AK_EMB_KERNEL_DIR", "AK_X9_KERNEL_DIR", "AK_NVFP4P_EXT_DIR", "AK_NVFP4P_W2D_EXT_DIR",
            "AK_NVFP4P_NORMCAST_EXT_DIR", "AK_NVFP4P_RHT8_EXT_DIR", "AK_NVFP4P_BX_EXT_DIR", "AK_FP8CS_EXT_DIR",
            "AK_FP4G_EXT_DIR", "AK_LMH_DIR")
CLUSTER = re.compile(r"/(?:home|shared|vast)/[A-Za-z0-9_.-]")
PTH_WORDS = ("ak_l8b", "firmus_ak", "auto_kernel")


def _site_dirs():
    dirs = list(site.getsitepackages())
    user = site.getusersitepackages()
    if isinstance(user, str):
        dirs.append(user)
    dirs += [p for p in sys.path if p.endswith(("site-packages", "dist-packages"))]
    seen, out = set(), []
    for d in dirs:
        r = os.path.realpath(d)
        if os.path.isdir(r) and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def check_pth(errors):
    for d in _site_dirs():
        for name in sorted(os.listdir(d)):
            if not name.endswith(".pth"):
                continue
            path = os.path.join(d, name)
            try:
                text = open(path, errors="replace").read()
            except OSError:
                text = ""
            if name.startswith("zz-ak") or any(w in name or w in text for w in PTH_WORDS):
                errors.append("a Firmus .pth loader is installed: %s" % path)


def _inside(path, prefix):
    rp, rprefix = os.path.realpath(path), os.path.realpath(prefix)
    return os.path.commonpath([rp, rprefix]) == rprefix


def check_env(prefix, exp, errors):
    env_dirs = exp.get("env_dirs", {})
    if set(env_dirs) != set(ENV_VARS):
        errors.append("EXPECTED_OUTPUTS.json env_dirs %s != the Dockerfile's %s" % (sorted(env_dirs), sorted(ENV_VARS)))
    for var in ENV_VARS:
        val = os.environ.get(var)
        if not val:
            errors.append("%s is not set" % var)
            continue
        if not _inside(val, prefix):
            errors.append("%s=%s is outside the prefix %s" % (var, val, prefix))
        want = os.path.join(prefix, env_dirs.get(var, "?"))
        if os.path.realpath(val) != os.path.realpath(want):
            errors.append("%s=%s, expected %s" % (var, val, want))
    for name, ext in sorted(exp.get("extensions", {}).items()):
        d = os.environ.get(ext.get("env_dir", ""), "")
        p = os.path.join(d, os.path.basename(ext["path"]))
        if not d or not os.path.isfile(p):
            errors.append("%s: %s=%s does not hold %s" % (name, ext.get("env_dir"), d, os.path.basename(ext["path"])))


def check_outputs(prefix, exp, errors):
    files = exp.get("files", [])
    if not files:
        errors.append("EXPECTED_OUTPUTS.json lists no files")
    for rel in files:
        p = os.path.join(prefix, rel)
        # the x9 tree's package markers (harness/, tier2/, x9_adamw/__init__.py) are empty in the validated tree
        empty_ok = os.path.basename(rel) == "__init__.py"
        if not os.path.isfile(p) or (os.path.getsize(p) == 0 and not empty_ok):
            errors.append("expected output missing or empty: %s" % p)
    return len(files)


def check_literals(errors):
    files = []
    spec = importlib.util.find_spec("firmus_ak")       # a top-level package: located, not imported
    locs = list(spec.submodule_search_locations or []) if spec is not None else []
    if not locs:
        errors.append("the firmus_ak package is not installed")
    for pkg in locs:
        for root, _dirs, names in os.walk(pkg):
            files += [os.path.join(root, n) for n in names if n.endswith((".py", ".json"))]
    for mod in MODULES:
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.origin:
            errors.append("patch module %s is not installed" % mod)
            continue
        files.append(spec.origin)
    for path in files:
        with open(path, errors="replace") as f:
            for n, line in enumerate(f, 1):
                if CLUSTER.search(line):
                    errors.append("cluster path in %s:%d: %s" % (path, n, line.strip()[:120]))
    return len(files)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefix", default=os.environ.get("FIRMUS_AK_PREFIX", "/opt/firmus/ak"))
    args = ap.parse_args(argv)
    prefix = os.path.abspath(args.prefix)
    errors = []
    src = os.path.join(prefix, "EXPECTED_OUTPUTS.json")
    try:
        with open(src) as f:
            exp = json.load(f)
    except (OSError, ValueError) as e:
        exp = {}
        errors.append("cannot read %s: %s" % (src, e))
    check_pth(errors)
    check_env(prefix, exp, errors)
    n_out = check_outputs(prefix, exp, errors)
    n_files = check_literals(errors)
    if errors:
        for e in errors:
            print("check_install: FAIL: " + e, file=sys.stderr)
        print("check_install: FAILED (%d problems)" % len(errors), file=sys.stderr)
        return 1
    print("check_install: PASSED: no Firmus .pth; %d AK_*_DIR variables inside %s; %d expected outputs present (%s); "
          "%d Firmus Python files free of cluster paths" % (len(ENV_VARS), prefix, n_out, src, n_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
