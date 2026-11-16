#!/usr/bin/env python3
"""build_all.py -- builds the Firmus L4 auto_kernel extensions from ./csrc and lays out the kernel files under
${FIRMUS_AK_PREFIX} (default /opt/firmus/ak).  Run by the Dockerfile (section 5); nothing prebuilt is copied.

    python3 build_all.py --manifest build_manifest.json --out ${FIRMUS_AK_PREFIX}/lib      build + lay out + verify
    python3 build_all.py --manifest build_manifest.json --out ${FIRMUS_AK_PREFIX}/lib --verify-only
    python3 build_all.py --manifest build_manifest.json --dry-run [--only ak_l8b_rht7,...]   print the plan, build nothing
    python3 build_all.py --manifest build_manifest.json --write-expected                    (re)write EXPECTED_OUTPUTS.json

Every extension is built the way its validated module was (build_manifest.json, field "evidence"):
torch.utils.cpp_extension.load() with that module's name, sources, include paths, flags and TORCH_CUDA_ARCH_LIST,
in a child process of its own, from a flat staging directory holding its sources and headers (the dev-directory layout
the validated build compiled).  load() also imports the built module, so a module that does not import fails here.
Before anything is compiled, the generated headers are regenerated from their generators -- two of them from TE's own
kernel text in TE_SRC (/workspace/transformerengine) -- and must equal the shipped bytes (regen_checks).

Fail loud: any failed regeneration, compile, import, copy or check exits non-zero and the image build stops.

--verify-only: every expected output present and non-empty; every Python kernel / data file byte-equal to its source
in this tree (SOURCES.sha256); every extension an ELF shared object exporting PyInit_<name>, with a cubin for the
manifest's arch (cuobjdump --list-elf) unless it is host-only; one line per output with its sha256; writes
${FIRMUS_AK_PREFIX}/BUILD_LOG.json (flags, evidence, shas, cubins) and ${FIRMUS_AK_PREFIX}/EXPECTED_OUTPUTS.json.

FIRMUS_AK_BUILD_NOIMPORT=1 builds with torch's ninja writer without importing the module (a fallback for a build
host where importing a CUDA extension is impossible); the default is load(), the validated mechanism.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "csrc")


def die(msg):
    print("[build_all] FAILED: %s" % msg, flush=True)
    sys.exit(1)


def say(msg):
    print("[build_all] %s" % msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_manifest(path):
    with open(path) as f:
        m = json.load(f)
    if m.get("schema") != "firmus-ak-build-manifest/1":
        die("%s: unknown schema %r" % (path, m.get("schema")))
    names = [e["name"] for e in m["extensions"]]
    if len(names) != len(set(names)):
        die("duplicate extension names in the manifest")
    for e in m["extensions"]:
        for k in ("name", "sources", "headers", "include_src_dir", "te_include", "extra_cflags", "extra_cuda_cflags",
                  "extra_ldflags", "with_cuda", "arch", "cubin", "out_dir", "output", "env_dir", "evidence",
                  "flags_status"):
            if k not in e:
                die("extension %s: field %s missing" % (e.get("name"), k))
        if e["output"] != e["name"] + ".so":
            die("extension %s: output %s is not <name>.so (the import name must equal the file name)"
                % (e["name"], e["output"]))
        if e["out_dir"] not in m["env_dirs"].values():
            die("extension %s: out_dir %s is no AK_*_DIR of env_dirs" % (e["name"], e["out_dir"]))
        if m["env_dirs"].get(e["env_dir"]) != e["out_dir"]:
            die("extension %s: %s does not point at %s" % (e["name"], e["env_dir"], e["out_dir"]))
        for s in e["sources"] + e["headers"]:
            if not os.path.isfile(os.path.join(CSRC, s)):
                die("extension %s: source %s not in %s" % (e["name"], s, CSRC))
    for x in m["layout"]:
        if not os.path.isfile(os.path.join(HERE, x["from"])):
            die("layout: %s missing" % x["from"])
        if m["env_dirs"].get(x["env_dir"]) != x["to"].split("/")[0]:
            die("layout: %s is not under %s's directory" % (x["to"], x["env_dir"]))
    return m


def expected_outputs(m):
    files = sorted(["%s/%s" % (e["out_dir"], e["output"]) for e in m["extensions"]] + [x["to"] for x in m["layout"]])
    return {
        "schema": "firmus-ak-expected-outputs/1",
        "prefix_env": "FIRMUS_AK_PREFIX",
        "files": files,
        "extensions": {e["name"]: {"path": "%s/%s" % (e["out_dir"], e["output"]), "cubin": e["cubin"],
                                   "env_dir": e["env_dir"]} for e in m["extensions"]},
        "env_dirs": m["env_dirs"],
    }


def te_paths(m):
    te_src = os.environ.get(m["te_src_env"], m["te_src_default"])
    return te_src


def te_lib():
    """the installed transformer_engine package directory and the one libtransformer_engine*.so in it, found without
    importing transformer_engine (as test_w2d.py's os.listdir of the package directory)"""
    import importlib.util
    spec = importlib.util.find_spec("transformer_engine")
    if spec is None or not spec.submodule_search_locations:
        die("transformer_engine is not installed (needed by ak_l8b_w2d7's link)")
    d = list(spec.submodule_search_locations)[0]
    libs = sorted(f for f in os.listdir(d) if f.startswith("libtransformer_engine") and f.endswith(".so"))
    if len(libs) != 1:
        die("expected exactly one libtransformer_engine*.so in %s, found %s" % (d, libs))
    return d, libs[0]


def expand(e, m, te_src, stage, need_te_lib):
    """the load() keyword arguments of one extension, every placeholder resolved"""
    def flags(lst, subst):
        if lst is None:
            return None
        out = []
        for f in lst:
            if f == "@half_flags":
                out += m["half_flags"]
            else:
                out.append(f.format(**subst))
        return out
    subst = {}
    if need_te_lib and any("{TE_LIB" in f for f in (e["extra_ldflags"] or [])):
        subst["TE_LIB"], subst["TE_LIBNAME"] = te_lib()
    elif any("{TE_LIB" in f for f in (e["extra_ldflags"] or [])):
        subst["TE_LIB"], subst["TE_LIBNAME"] = "<TE_LIB>", "<libtransformer_engine.so>"
    inc = ([stage] if e["include_src_dir"] else []) + [os.path.join(te_src, p) for p in e["te_include"]]
    kw = {
        "name": e["name"],
        "sources": [os.path.join(stage, os.path.basename(s)) for s in e["sources"]],
        "extra_include_paths": inc or None,
        "extra_cflags": flags(e["extra_cflags"], subst),
        "extra_cuda_cflags": flags(e["extra_cuda_cflags"], subst),
        "extra_ldflags": flags(e["extra_ldflags"], subst),
    }
    if e["with_cuda"] is not None:
        kw["with_cuda"] = e["with_cuda"]
    return kw


def stage_sources(e, stage):
    os.makedirs(stage, exist_ok=True)
    seen = {}
    for s in e["sources"] + e["headers"]:
        b = os.path.basename(s)
        if b in seen:
            die("extension %s: two staged files named %s (%s, %s)" % (e["name"], b, seen[b], s))
        seen[b] = s
        shutil.copyfile(os.path.join(CSRC, s), os.path.join(stage, b))


def regen_checks(m, te_src, dry):
    """re-run the generators of the generated headers and compare with the shipped bytes"""
    results = []
    for c in m["regen_checks"]:
        tmp = tempfile.mkdtemp(prefix="firmus_ak_regen_")
        subst = {"TE_SRC": te_src, "TMP": tmp, "CSRC": CSRC}
        for step in c["steps"]:
            gen = os.path.join(CSRC, step[0])
            args = [a.format(**subst) for a in step[1:]]
            for a in args[:-1]:
                if not os.path.isfile(a):
                    die("regen '%s': input %s missing" % (c["what"], a))
            # -I: the generator sees neither this directory nor the environment's site customisations; cwd = an empty
            # temporary directory, so it can read only the files named on its command line
            r = subprocess.run([sys.executable, "-I", gen] + args, cwd=tmp, capture_output=True, text=True)
            if r.returncode != 0:
                die("regen '%s': %s exited %d\n%s%s" % (c["what"], step[0], r.returncode, r.stdout, r.stderr))
        for got, want in c["compare"]:
            got = got.format(**subst)
            want_p = os.path.join(CSRC, want)
            a, b = sha256(got), sha256(want_p)
            ok = a == b
            say("regen %-28s %s  %s" % (want, "equal" if ok else "DIFFERS", a))
            results.append({"file": want, "regenerated_sha256": a, "shipped_sha256": b, "equal": ok,
                            "what": c["what"]})
            if not ok:
                die("regen '%s': %s regenerated as %s, shipped %s -- the generator input changed (for the TE-derived "
                    "files: TE_SRC is not the TE source the kernels were derived from)" % (c["what"], want, a, b))
        shutil.rmtree(tmp, ignore_errors=True)
    return results


# --- the child: one extension, one process --------------------------------------------------------------------------
def child(args):
    with open(args.child_kw) as f:
        job = json.load(f)
    kw, arch = job["kw"], job["arch"]
    if arch is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    os.makedirs(job["build_dir"], exist_ok=True)
    t0 = time.time()
    if os.environ.get("FIRMUS_AK_BUILD_NOIMPORT") == "1":
        import inspect
        from torch.utils import cpp_extension as ce
        fn = ce._write_ninja_file_and_build_library   # torch-private: the build half of load()
        srcs = kw["sources"]
        with_cuda = kw.get("with_cuda")
        if with_cuda is None:
            with_cuda = any(ce._is_cuda_file(s) for s in srcs)
        full = {"name": kw["name"], "sources": srcs, "extra_cflags": kw["extra_cflags"] or [],
                "extra_cuda_cflags": kw["extra_cuda_cflags"] or [], "extra_sycl_cflags": [],
                "extra_ldflags": kw["extra_ldflags"] or [], "extra_include_paths": kw["extra_include_paths"] or [],
                "build_directory": job["build_dir"], "verbose": True, "with_cuda": with_cuda, "with_sycl": False,
                "is_standalone": False}
        accepted = inspect.signature(fn).parameters
        fn(**{k: v for k, v in full.items() if k in accepted})
        so = os.path.join(job["build_dir"], kw["name"] + ".so")
    else:
        from torch.utils.cpp_extension import load
        mod = load(build_directory=job["build_dir"], verbose=True, **kw)
        so = mod.__file__
        if getattr(mod, "__name__", None) != kw["name"]:
            die("%s: load() returned module %r" % (kw["name"], getattr(mod, "__name__", None)))
    if not os.path.isfile(so):
        die("%s: no %s after the build" % (kw["name"], so))
    if os.path.basename(so) != kw["name"] + ".so":
        die("%s: built %s, not %s.so (torch's JIT versioning renamed it: the build directory was not fresh)"
            % (kw["name"], so, kw["name"]))
    print("FIRMUS_AK_CHILD_RESULT " + json.dumps({"so": so, "sha256": sha256(so), "seconds": round(time.time() - t0, 1)}),
          flush=True)


# --- ELF / cubin checks ----------------------------------------------------------------------------------------------
def elf_dynamic_symbols(path):
    """names of the defined dynamic symbols of an ELF64 little-endian shared object (no external tools)"""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        die("%s: not an ELF64 little-endian file" % path)
    e_type = struct.unpack_from("<H", data, 16)[0]
    if e_type != 3:
        die("%s: ELF type %d, not a shared object" % (path, e_type))
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    secs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        (sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign,
         sh_entsize) = struct.unpack_from("<IIQQQQIIQQ", data, off)
        secs.append((sh_type, sh_offset, sh_size, sh_link, sh_entsize))
    names = set()
    for sh_type, off, size, link, entsize in secs:
        if sh_type != 11:          # SHT_DYNSYM
            continue
        stroff = secs[link][1]
        for j in range(size // (entsize or 24)):
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", data, off + j * 24)
            if st_shndx == 0 or st_name == 0:
                continue
            end = data.index(b"\0", stroff + st_name)
            names.add(data[stroff + st_name:end].decode("ascii", "replace"))
    return names


def cubins(path):
    exe = shutil.which("cuobjdump") or (os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "cuobjdump"))
    if not os.path.isfile(exe):
        die("cuobjdump not found (needed to check %s's device code)" % path)
    r = subprocess.run([exe, "--list-elf", path], capture_output=True, text=True)
    if r.returncode != 0:
        die("cuobjdump --list-elf %s exited %d: %s" % (path, r.returncode, r.stderr.strip()))
    archs = sorted({tok.split(".")[-2] for line in r.stdout.splitlines() for tok in line.split()
                    if tok.endswith(".cubin") and tok.count(".") >= 2})
    return archs


# --- main ------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=os.path.join(HERE, "build_manifest.json"))
    ap.add_argument("--out", help="the lib directory (default ${FIRMUS_AK_PREFIX}/lib)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="comma-separated extension names (with --dry-run, or a build for a test; never "
                                   "with --verify-only)")
    ap.add_argument("--build-root", default=os.environ.get("FIRMUS_AK_BUILD_ROOT", "/tmp/firmus_ak_build"))
    ap.add_argument("--write-expected", action="store_true", help="write ./EXPECTED_OUTPUTS.json from the manifest")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--child-kw", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child(args)

    m = load_manifest(args.manifest)
    exp = expected_outputs(m)
    if args.write_expected:
        with open(os.path.join(HERE, "EXPECTED_OUTPUTS.json"), "w") as f:
            json.dump(exp, f, indent=1)
            f.write("\n")
        say("wrote %s (%d files)" % (os.path.join(HERE, "EXPECTED_OUTPUTS.json"), len(exp["files"])))
        return
    shipped_exp = os.path.join(HERE, "EXPECTED_OUTPUTS.json")
    if os.path.isfile(shipped_exp):
        with open(shipped_exp) as f:
            if json.load(f) != exp:
                die("EXPECTED_OUTPUTS.json differs from the manifest (run --write-expected and review)")

    prefix = os.path.abspath(os.environ.get("FIRMUS_AK_PREFIX", "/opt/firmus/ak"))
    lib = os.path.abspath(args.out or os.path.join(prefix, "lib"))
    if lib != os.path.join(prefix, m["env_dirs"]["AK_NVFP4P_EXT_DIR"]):
        die("--out %s is not ${FIRMUS_AK_PREFIX}/lib (%s)" % (lib, os.path.join(prefix, "lib")))
    te_src = te_paths(m)
    only = set(args.only.split(",")) if args.only else None
    exts = [e for e in m["extensions"] if only is None or e["name"] in only]
    if only and len(exts) != len(only):
        die("--only names unknown extensions: %s" % sorted(only - {e["name"] for e in m["extensions"]}))
    if args.verify_only and only:
        die("--verify-only checks every output; --only is not allowed with it")
    if not args.dry_run and not args.verify_only and only:
        say("WARNING: --only builds a subset; --verify-only will fail until every extension is built")

    if args.verify_only:
        return verify(m, exp, prefix)

    say("prefix %s; TE_SRC %s; manifest %s (sha256 %s)" % (prefix, te_src, args.manifest, sha256(args.manifest)))
    if not os.path.isdir(os.path.join(te_src, "transformer_engine")):
        die("TE_SRC %s has no transformer_engine/ (the TE source tree of section 2 must be kept)" % te_src)
    regen = regen_checks(m, te_src, args.dry_run)

    if args.dry_run:
        for e in exts:
            stage = os.path.join(args.build_root, e["name"], "src")
            kw = expand(e, m, te_src, stage, need_te_lib=False)
            say("---- %s (%s; %s) -> ${P}/%s/%s" % (e["name"], e["id"], e["flags_status"], e["out_dir"], e["output"]))
            say("  stage %s <- %s" % (stage, " ".join(e["sources"] + e["headers"])))
            say("  TORCH_CUDA_ARCH_LIST=%s MAX_JOBS=%s" % (e["arch"], os.environ.get("MAX_JOBS", "(torch default)")))
            say("  torch.utils.cpp_extension.load(%s, build_directory=%r)" % (
                ", ".join("%s=%r" % (k, v) for k, v in kw.items()), os.path.join(args.build_root, e["name"], "build")))
            say("  nvcc (as load() writes it): %s" % nvcc_line(e, kw))
        for x in m["layout"]:
            say("layout %s -> ${P}/%s" % (x["from"], x["to"]))
        say("dry run: %d extensions, %d layout files, %d regen checks equal" % (len(exts), len(m["layout"]),
                                                                              sum(r["equal"] for r in regen)))
        return

    # build every extension, one child process each
    records = {}
    for e in exts:
        root = os.path.join(args.build_root, e["name"])
        if os.path.exists(root):
            die("build root %s exists: use a fresh --build-root (a stale ninja tree or JIT version would be reused)"
                % root)
        stage, bdir = os.path.join(root, "src"), os.path.join(root, "build")
        stage_sources(e, stage)
        kw = expand(e, m, te_src, stage, need_te_lib=True)
        kwfile = os.path.join(root, "kw.json")
        with open(kwfile, "w") as f:
            json.dump({"kw": kw, "arch": e["arch"], "build_dir": bdir}, f, indent=1)
        say("building %s (%s) TORCH_CUDA_ARCH_LIST=%s" % (e["name"], e["id"], e["arch"]))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", e["name"], "--child-kw", kwfile],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        sys.stdout.write(r.stdout)
        sys.stdout.flush()
        res = [l for l in r.stdout.splitlines() if l.startswith("FIRMUS_AK_CHILD_RESULT ")]
        if r.returncode != 0 or len(res) != 1:
            die("%s: build exited %d" % (e["name"], r.returncode))
        res = json.loads(res[0].split(" ", 1)[1])
        dst_dir = os.path.join(prefix, e["out_dir"])
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, e["output"])
        if os.path.exists(dst):
            die("%s exists already: the prefix must be empty before the build" % dst)
        shutil.copyfile(res["so"], dst)
        os.chmod(dst, 0o755)
        if sha256(dst) != res["sha256"]:
            die("%s: copy differs from the built file" % dst)
        records[e["name"]] = {"built": res["so"], "sha256": res["sha256"], "seconds": res["seconds"], "kw": kw,
                              "arch": e["arch"]}
        say("built %s %s in %.1f s -> %s" % (e["name"], res["sha256"], res["seconds"], dst))

    # the Python (Triton) kernels and the data files
    for x in m["layout"]:
        src, dst = os.path.join(HERE, x["from"]), os.path.join(prefix, x["to"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            die("%s exists already" % dst)
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o644)
    say("laid out %d kernel / data files" % len(m["layout"]))

    with open(os.path.join(prefix, "EXPECTED_OUTPUTS.json"), "w") as f:
        json.dump(exp, f, indent=1)
        f.write("\n")
    log = {"schema": "firmus-ak-build-log/1", "manifest_sha256": sha256(args.manifest), "te_src": te_src,
           "te_commit": _read(os.path.join("/", "TE_COMMIT_HASH.env")), "build_root": args.build_root,
           "max_jobs": os.environ.get("MAX_JOBS"), "noimport": os.environ.get("FIRMUS_AK_BUILD_NOIMPORT") == "1",
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "regen_checks": regen,
           "builds": records}
    with open(os.path.join(prefix, "BUILD_LOG.json"), "w") as f:
        json.dump(log, f, indent=1)
        f.write("\n")
    say("build done: %d extensions; run --verify-only next" % len(records))


def nvcc_line(e, kw):
    """an approximation of the nvcc command load() writes into build.ninja, for --dry-run only"""
    if not kw["sources"] or all(s.endswith(".cpp") for s in kw["sources"]):
        return "(host C++ only: c++ ... %s %s)" % (" ".join(kw["extra_cflags"] or []), " ".join(kw["extra_ldflags"] or []))
    gen = "-gencode=arch=compute_%s,code=sm_%s" % ((e["arch"].replace(".", ""),) * 2)
    inc = " ".join("-I" + p for p in (kw["extra_include_paths"] or []))
    return ("nvcc -DTORCH_EXTENSION_NAME=%s -DTORCH_API_INCLUDE_EXTENSION_H %s -isystem <torch> -isystem <cuda> "
            "-isystem <python> -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ "
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr %s "
            "--compiler-options '-fPIC' %s -std=<torch default> -c <src> ; link: %s"
            % (kw["name"], inc, gen, " ".join(kw["extra_cuda_cflags"] or []), " ".join(kw["extra_ldflags"] or [])))


def _read(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def verify(m, exp, prefix):
    say("verify %s" % prefix)
    if not os.path.isdir(prefix):
        die("%s does not exist: nothing was built" % prefix)
    src_sha = {}
    with open(os.path.join(HERE, "SOURCES.sha256")) as f:
        for line in f:
            if line.strip():
                h, p = line.split(None, 1)
                src_sha[p.strip()] = h
    bad = []
    files = {}
    for rel in exp["files"]:
        p = os.path.join(prefix, rel)
        if not os.path.isfile(p):
            bad.append("%s missing" % rel)
            continue
        if rel.endswith(".so") and os.path.getsize(p) == 0:
            bad.append("%s is empty" % rel)
            continue
        files[rel] = sha256(p)       # the layout files (three of them the x9 tree's empty __init__.py) are compared
                                     # with their sources below
    for x in m["layout"]:
        if x["to"] in files:
            want = src_sha.get(x["from"])
            if want is None:
                bad.append("%s not in SOURCES.sha256" % x["from"])
            elif files[x["to"]] != want or sha256(os.path.join(HERE, x["from"])) != want:
                bad.append("%s differs from its source %s (%s)" % (x["to"], x["from"], want))
            if x["to"].endswith(".py"):
                try:
                    compile(open(os.path.join(prefix, x["to"])).read(), x["to"], "exec")
                except SyntaxError as err:
                    bad.append("%s: %s" % (x["to"], err))
    ext_info = {}
    for e in m["extensions"]:
        rel = "%s/%s" % (e["out_dir"], e["output"])
        if rel not in files:
            continue
        p = os.path.join(prefix, rel)
        syms = elf_dynamic_symbols(p)
        if "PyInit_" + e["name"] not in syms:
            bad.append("%s exports no PyInit_%s" % (rel, e["name"]))
        arch_list = cubins(p) if e["cubin"] else []
        if e["cubin"] and e["cubin"] not in arch_list:
            bad.append("%s has no %s cubin (cubins: %s)" % (rel, e["cubin"], arch_list))
        if not e["cubin"] and arch_list:
            bad.append("%s is host-only in the manifest but carries cubins %s" % (rel, arch_list))
        ext_info[e["name"]] = {"path": rel, "sha256": files[rel], "cubins": arch_list, "arch": e["arch"],
                               "flags_status": e["flags_status"], "evidence": e["evidence"],
                               "sources": {s: sha256(os.path.join(CSRC, s)) for s in e["sources"] + e["headers"]}}
    for var, d in sorted(m["env_dirs"].items()):
        v = os.environ.get(var)
        if v is not None and os.path.abspath(v) != os.path.join(prefix, d):
            bad.append("%s=%s in the environment, but the kernels are laid out in %s" % (var, v, os.path.join(prefix, d)))
    extra = []
    for root, _, fs in os.walk(prefix):
        for f in fs:
            rel = os.path.relpath(os.path.join(root, f), prefix)
            if rel not in files and rel not in ("BUILD_LOG.json", "EXPECTED_OUTPUTS.json") \
                    and "__pycache__" not in rel:
                extra.append(rel)
    if extra:
        bad.append("unexpected files under the prefix: %s" % extra)
    for rel in sorted(files):
        print("%s  %s" % (files[rel], rel), flush=True)
    log_p = os.path.join(prefix, "BUILD_LOG.json")
    log = json.load(open(log_p)) if os.path.isfile(log_p) else {"schema": "firmus-ak-build-log/1"}
    log["verify"] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ok": not bad, "problems": bad,
                     "files": files, "extensions": ext_info}
    with open(log_p, "w") as f:
        json.dump(log, f, indent=1)
        f.write("\n")
    with open(os.path.join(prefix, "EXPECTED_OUTPUTS.json"), "w") as f:
        json.dump(exp, f, indent=1)
        f.write("\n")
    if bad:
        for b in bad:
            print("[build_all] VERIFY: %s" % b, flush=True)
        die("--verify-only: %d problem(s)" % len(bad))
    say("verify OK: %d files, %d extensions (PyInit and cubin checked), %d kernel files byte-equal to their sources"
        % (len(files), len(ext_info), len(m["layout"])))


if __name__ == "__main__":
    main()
