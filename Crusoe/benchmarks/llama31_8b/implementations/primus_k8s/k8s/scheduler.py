#!/usr/bin/env python3
"""Run the MLPerf v6.1 Llama scaling matrix to completion on one pool of MI355X nodes.

Each config (bench, nodes, gbs, mbs) needs TARGET passing runs (10 = one MLPerf result set).
  * gate: a config's first run runs alone; the rest of its set only starts after one PASS
    (a broken scale burns one run, not ten). Two FAILs before any PASS blocks the config.
  * packing: runs get disjoint node sets (label <SELKEY>=<slot>) and run concurrently;
    configs earlier in the list have priority, smaller ones backfill unless a higher-priority
    run has been waiting > RESERVE_S for nodes.
  * every PASS: the full rank-0 log (framework output + MLLOG) ->
    results/<N>xMI355X-288GB_2xEPYC-9575F/<bench>/result_<i>.txt, then the 6.1.0 compliance checker.
  * node hygiene: a node that is part of failed runs in 2 different configs is quarantined
    (never scheduled again); failures before run_start (FAIL_PRESTART) blame no node.
State is rebuilt from runs.tsv on restart, so the scheduler can be killed and re-run.
Every run in OUT is part of the record: declare a fresh OUT per submission round and never
cherry-pick runs from it (packaging applies the M>N consecutive-window rule to all of them).

Benchmarks: llama31_8b, llama2_70b_lora (run-llama-mlperf.sh) and flux1 (../bundle2/run-flux-compliance.sh).
A config with a run in flight never holds a node reservation, so smaller configs backfill idle nodes.

  CTX=<ctx> NS=<ns> OUT=<dir> REGISTRY=<registry> [MANUAL_GC=1] [BENCH_ONLY=<bench>]
    [SETS_ONLY="<nodes>:<bench> ..."] python3 scheduler.py [--dry-run]
"""
import json, os, re, subprocess, sys, time, datetime, collections

CTX, NS = os.environ["CTX"], os.environ["NS"]
OUT = os.environ["OUT"]
HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.environ.get("REGISTRY", "")
IMAGES = {"llama31_8b": f"{REG}/llama31-8b-libionic:v61-a77-r2",
          "llama2_70b_lora": f"{REG}/llama2-70b-lora-libionic:v61-a77-r2",
          "flux1": os.environ.get("FLUX_IMAGE", f"{REG}/flux1-libionic:bundle2-v1.0")}
# Per-benchmark runner: (directory, script, files snapshotted per run, rank-0 log path on the shared volume).
# Every runner prints "VERDICT <PASS|FAIL|FAIL_PRESTART> <job> bench= nn= gbs= seed= ttt_s= last_eval= job= run=".
BUNDLE2 = os.path.join(HERE, "..", "bundle2")
RUNNERS = {
    "llama31_8b": (HERE, "run-llama-mlperf.sh", ("run-llama-mlperf.sh", "launch-llama.sh", "train-llama.template.yaml"),
                   "/shared/llama-v61/logs/{run}/rank0.log"),
    "flux1": (BUNDLE2, "run-flux-compliance.sh", ("run-flux-compliance.sh", "train-32node-compliance.template.yaml"),
              "/shared/logs/{run}/rank0.log"),
}
RUNNERS["llama2_70b_lora"] = RUNNERS["llama31_8b"]
FLUX_SET = {1024: "C", 2048: "A", 4096: "B"}   # run-flux-compliance.sh set letter per RCP batch-size row
SELKEY = "mlperf.crusoe.ai/llama"
TARGET = int(os.environ.get("TARGET", "10"))
RESERVE_S = int(os.environ.get("RESERVE_S", "900"))
EXCLUDE = set(os.environ.get("EXCLUDE", "").split())
DRY = "--dry-run" in sys.argv
# priority order
CONFIGS = [   # (bench, nodes, gbs, mbs, cp) in priority order -- longest node-hours first so the
              # big jobs reserve nodes and single-node runs backfill the gaps.
              # LoRA scales by CP (keeps GBS small), 8B by DP.
    ("llama31_8b", 16, 128, 1, 1), ("llama31_8b", 1, 32, 2, 1),
    ("llama31_8b", 8, 64, 1, 1), ("llama31_8b", 4, 32, 1, 1),
    # 32 nodes: the largest RCP batch (128) on 256 GPUs needs model parallelism -> CP2 x DP128.
    # (TP2 hung at init with MXFP4 + sequence parallel; CP2 trained clean in a 1-node test.)
    ("llama31_8b", 32, 128, 1, 2),
    ("llama2_70b_lora", 16, 16, 1, 8), ("llama2_70b_lora", 8, 8, 1, 8),
    ("llama2_70b_lora", 4, 8, 1, 4), ("llama2_70b_lora", 1, 8, 1, 1),
    # Flux.1: (bench, nodes, gbs, mbs=gbs/gpus, 1). Largest per-GPU batch that maps onto an RCP row;
    # per-GPU throughput falls steeply with MBS (MBS 32 ~50, 16 ~31, 8 ~17 samples/s/GPU).
    ("flux1", 16, 2048, 16, 1), ("flux1", 8, 2048, 32, 1), ("flux1", 4, 1024, 32, 1),
    # 32 nodes: RCP row 4096 (set B: LR 4e-4, warmup 400), MBS 16 -- runs under the flux1 RCP bypass
    ("flux1", 32, 4096, 16, 1),
]
# optional: run only one benchmark's configs (e.g. BENCH_ONLY=llama31_8b)
if os.environ.get("BENCH_ONLY"):
    CONFIGS = [c for c in CONFIGS if c[0] == os.environ["BENCH_ONLY"]]
# optional: run only these sets, "<nodes>:<bench> ..." (e.g. SETS_ONLY="4:llama2_70b_lora")
if os.environ.get("SETS_ONLY"):
    CONFIGS = [c for c in CONFIGS if f"{c[1]}:{c[0]}" in os.environ["SETS_ONLY"].split()]
TIMEOUT = {"llama31_8b": {1: 4 * 3600}, "llama2_70b_lora": {}, "flux1": {4: 4 * 3600, 8: 3 * 3600, 16: 2 * 3600, 32: 2 * 3600}}
os.makedirs(f"{OUT}/runlogs", exist_ok=True)
RUNS_TSV = f"{OUT}/runs.tsv"
LOG = open(f"{OUT}/scheduler.log", "a", buffering=1)

def log(msg):
    line = f"[{datetime.datetime.utcnow():%H:%M:%SZ}] {msg}"
    print(line, flush=True); LOG.write(line + "\n")

def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}: {r.stderr}")
    return r.stdout

def k(args):
    return sh(f"kubectl --context {CTX} {args}")

def pool():
    d = json.loads(k("get nodes -l amd.com/gpu.product-name=AMD_Instinct_MI355_OAM -o json") or '{"items":[]}')
    ok = []
    for n in d["items"]:
        name = n["metadata"]["name"].split(".")[0]
        ready = any(c["type"] == "Ready" and c["status"] == "True" for c in n["status"]["conditions"])
        held = SELKEY in n["metadata"].get("labels", {})    # someone else's run (or ours, filtered by busy)
        if ready and not n["spec"].get("unschedulable") and name not in EXCLUDE and not held:
            ok.append(name)
    return sorted(ok, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])

def label(nodes, val):
    for n in nodes:
        v = f"{SELKEY}={val}" if val else f"{SELKEY}-"
        k(f"label node {n}.us-east2-a.compute.internal {v} --overwrite")

# ---- persistent state ----
def load_runs():
    rows = []
    if os.path.exists(RUNS_TSV):
        for line in open(RUNS_TSV):
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            rows.append(dict(zip(["ts", "bench", "nn", "gbs", "mbs", "seed", "verdict", "ttt", "eval", "job", "run", "nodes", "comp", "result", "cp"], f)))
    return rows

def append_run(r):
    new = not os.path.exists(RUNS_TSV)
    with open(RUNS_TSV, "a") as f:
        if new:
            f.write("#ts\tbench\tnn\tgbs\tmbs\tseed\tverdict\tttt_s\tlast_eval\tjob\trun\tnodes\tcompliance\tresult_file\tcp\n")
        f.write("\t".join(str(r[x]) for x in ["ts", "bench", "nn", "gbs", "mbs", "seed", "verdict", "ttt", "eval", "job", "run", "nodes", "comp", "result", "cp"]) + "\n")

def sysname(nn):
    return f"{nn}xMI355X-288GB_2xEPYC-9575F"

def collect(bench, nn, run, idx):
    """full rank-0 log -> result_<idx>.txt; run the 6.1.0 compliance checker on it."""
    d = f"{OUT}/results/{sysname(nn)}/{bench}"; os.makedirs(d, exist_ok=True)
    raw = k(f"-n {NS} exec log-reader -- cat {RUNNERS[bench][3].format(run=run)}")
    # MLPerf training rules ("Result files"): result_*.txt must hold the full framework log
    # plus all MLLOG lines -- so keep the whole rank-0 log (carriage returns -> newlines).
    path = f"{d}/result_{idx}.txt"
    open(path, "w").write(raw.replace("\r", "\n"))
    out = sh(f"python3 -m mlperf_logging.compliance_checker --usage training --ruleset 6.1.0 {path} 2>&1")
    ok = "SUCCESS" in out
    open(f"{path}.compliance", "w").write(out)
    return ("PASS" if ok else "FAIL"), path

def reconcile(stat):
    """Record verdicts that runners wrote while no scheduler was reaping (e.g. during a restart)."""
    seen = {r["job"] for r in load_runs()}
    for f in sorted(os.listdir(f"{OUT}/runlogs")):
        if not f.endswith(".out"):
            continue
        for line in open(f"{OUT}/runlogs/{f}"):
            if line[1:10] < os.environ.get("RECONCILE_SINCE", "99:99:99"):   # "[HH:MM:SSZ]"; default: none
                continue
            m = re.search(r"VERDICT (PASS|FAIL\S*) (\S+) bench=(\S+) nn=(\d+) gbs=(\d+) seed=(\d+) ttt_s=(\S+) last_eval=(\S+) job=\S* run=(\S+)", line)
            if not m or m.group(2) in seen:
                continue
            verdict, job, bench, nn, gbs, seed, ttt, ev, run = m.groups()
            key = next((c for c in CONFIGS if c[0] == bench and c[1] == int(nn) and c[2] == int(gbs)), None)
            if not key:
                continue
            comp, res = "-", "-"
            if verdict == "PASS":
                comp, res = collect(bench, int(nn), run, stat[key]["pass"])
                stat[key]["pass"] += 1
            else:
                stat[key]["fail"] += 1
            append_run(dict(ts=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), bench=bench, nn=nn, gbs=gbs,
                            mbs=key[3], seed=seed, verdict=verdict, ttt=ttt, eval=ev, job=job, run=run, nodes="",
                            comp=comp, result=os.path.basename(res), cp=key[4]))
            seen.add(job)
            log(f"RECONCILE {verdict} {bench} {nn}n seed={seed} ttt={ttt}s comp={comp}")

def adopt():
    """Re-attach to Llama jobs still in the cluster (launched by a previous scheduler)."""
    out = {}
    d = json.loads(k(f"-n {NS} get jobs -l app=llama-v61 -o json") or '{"items":[]}')
    for j in d["items"]:
        # running OR finished-but-unreaped: the runner deletes a job after its verdict, so any
        # Llama job still present has not been recorded yet
        c = j["spec"]["template"]["spec"]["containers"][0]
        env = {e["name"]: e.get("value") for e in c.get("env", [])}
        if "BENCH" not in env:
            continue
        slot = j["spec"]["template"]["spec"]["nodeSelector"].get(SELKEY)
        key = (env["BENCH"], int(env["NN"]), int(env["GBS"]), int(env["MBS"]), int(env.get("CP") or 1))
        if key not in CONFIGS or not slot:
            continue
        nodes = [n["metadata"]["name"].split(".")[0] for n in json.loads(k(f"get nodes -l {SELKEY}={slot} -o json"))["items"]]
        name = j["metadata"]["name"]
        o = f"{OUT}/runlogs/{env['BENCH']}-{env['NN']}n-seed{env['SEED']}.out"
        snap = f"{OUT}/runlogs/.run-{slot}"; os.makedirs(snap, exist_ok=True)
        for f in ("run-llama-mlperf.sh", "launch-llama.sh", "train-llama.template.yaml"):
            sh(f"cp {HERE}/{f} {snap}/")
        e2 = dict(os.environ, IMAGE=IMAGES[env["BENCH"]], BENCH=env["BENCH"], NN=env["NN"], GBS=env["GBS"], MBS=env["MBS"],
                  SEED=env["SEED"], ADOPT_J=name, ADOPT_RUN=env["RUN_NAME"],
                  TIMEOUT=str(TIMEOUT.get(env["BENCH"], {}).get(int(env["NN"]), 2 * 3600)))
        p = subprocess.Popen(["bash", f"{snap}/run-llama-mlperf.sh"], env=e2, stdout=open(o, "a"), stderr=subprocess.STDOUT)
        out[slot] = dict(proc=p, key=key, nodes=nodes, seed=int(env["SEED"]), out=o, start=time.time())
        log(f"ADOPT {name} slot={slot} nodes={','.join(nodes)}")
    # Flux jobs (bundle2 runner) carry no app label; recognise them by name flux-<N>n-<set>-s<seed>
    d = json.loads(k(f"-n {NS} get jobs -o json") or '{"items":[]}')
    for j in d["items"]:
        name = j["metadata"]["name"]
        m = re.fullmatch(r"flux-(\d+)n-([abc])-s(\d+)", name)
        if not m:
            continue
        nn, letter, seed = int(m.group(1)), m.group(2).upper(), int(m.group(3))
        gbs = {v: g for g, v in FLUX_SET.items()}[letter]
        key = ("flux1", nn, gbs, gbs // (nn * 8), 1)
        slot = j["spec"]["template"]["spec"]["nodeSelector"].get(SELKEY)
        if key not in CONFIGS or not slot:
            continue
        env = {e["name"]: e.get("value") for e in j["spec"]["template"]["spec"]["containers"][0].get("env", [])}
        nodes = [n["metadata"]["name"].split(".")[0] for n in json.loads(k(f"get nodes -l {SELKEY}={slot} -o json"))["items"]]
        o = f"{OUT}/runlogs/flux1-{nn}n-seed{seed}.out"
        rdir, script, files, _ = RUNNERS["flux1"]
        snap = f"{OUT}/runlogs/.run-{slot}"; os.makedirs(snap, exist_ok=True)
        for f in files:
            sh(f"cp {rdir}/{f} {snap}/")
        e2 = dict(os.environ, IMAGE=IMAGES["flux1"], N=str(nn), SELKEY=SELKEY, SELVAL=slot, PULL_SECRET="ccr",
                  ADOPT_J=name, ADOPT_RUN=env["RUN_NAME"], TIMEOUT=str(TIMEOUT["flux1"].get(nn, 2 * 3600)))
        p = subprocess.Popen(["bash", f"{snap}/{script}", letter, str(seed)], env=e2, stdout=open(o, "a"), stderr=subprocess.STDOUT)
        out[slot] = dict(proc=p, key=key, nodes=nodes, seed=seed, out=o, start=time.time())
        log(f"ADOPT {name} slot={slot} nodes={','.join(nodes)}")
    return out

def main():
    rows = load_runs()
    stat = collections.defaultdict(lambda: {"pass": 0, "fail": 0})
    nodefail = collections.defaultdict(set)   # node -> configs it failed in (2 distinct configs = node fault)
    for r in rows:
        key = (r["bench"], int(r["nn"]), int(r["gbs"]), int(r["mbs"]), int(r.get("cp") or 1))
        stat[key]["pass" if r["verdict"] == "PASS" else "fail"] += 1
        if r["verdict"] == "FAIL":
            for n in r["nodes"].split(","):
                nodefail[n].add(key)
    quarantined = {n for n, c in nodefail.items() if len(c) >= 2}
    # nodes cleared by a follow-up diagnostic (the failure was attributed to another node)
    quarantined -= set(os.environ.get("QUARANTINE_CLEAR", "").split())
    running = {}          # slot -> dict(proc, key, nodes, seed, out, start)
    waiting_since = {}
    last_launch = {}       # config -> time of its most recent launch (or adoption)
    slot_n = 0
    reconcile(stat)
    running.update(adopt())
    for ru in running.values():
        last_launch[ru["key"]] = time.time()
    log(f"scheduler start: target={TARGET} configs={len(CONFIGS)} quarantined={sorted(quarantined)}")
    while True:
        # ---- reap ----
        for slot, ru in list(running.items()):
            if ru["proc"].poll() is None:
                continue
            txt = open(ru["out"]).read()
            m = re.search(r"VERDICT (PASS|FAIL\S*) (\S+) .*ttt_s=(\S+) last_eval=(\S+) job=\S* run=(\S+)", txt)
            verdict, job, ttt, ev, run = (m.groups() if m else ("FAIL", "?", "NA", "NA", "?"))
            bench, nn, gbs, mbs, cp = ru["key"]
            comp, res = "-", "-"
            if verdict == "PASS":
                idx = stat[ru["key"]]["pass"]
                comp, res = collect(bench, nn, run, idx)
                stat[ru["key"]]["pass"] += 1
            elif verdict == "FAIL_PRESTART":
                stat[ru["key"]]["fail"] += 1             # harness/config problem: don't blame nodes
            else:
                stat[ru["key"]]["fail"] += 1
                for n in ru["nodes"]:
                    nodefail[n].add(ru["key"])
                    if len(nodefail[n]) >= 2 and n not in quarantined:
                        quarantined.add(n); log(f"ALERT quarantine {n} (failed in {len(nodefail[n])} distinct configs)")
            append_run(dict(ts=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), bench=bench, nn=nn, gbs=gbs,
                            mbs=mbs, seed=ru["seed"], verdict=verdict, ttt=ttt, eval=ev, job=job, run=run,
                            nodes=",".join(ru["nodes"]), comp=comp, result=os.path.basename(res), cp=cp))
            log(f"DONE {verdict} {bench} {nn}n gbs={gbs} seed={ru['seed']} ttt={ttt}s eval={ev} comp={comp} "
                f"[{stat[ru['key']]['pass']}/{TARGET} pass, {stat[ru['key']]['fail']} fail]")
            # unlabel only nodes still carrying THIS run's slot (an operator may have re-labelled
            # a node to fence it off, e.g. hold-<reason>; that label must survive the reap)
            k(f"label nodes -l {SELKEY}={slot} {SELKEY}-")
            del running[slot]
        # ---- schedule ----
        busy = {n for ru in running.values() for n in ru["nodes"]}
        free = [n for n in pool() if n not in busy and n not in quarantined]
        launched_any = False
        # Reservation: the config that has waited longest beyond RESERVE_S is the head. Others may
        # only launch if they leave enough free nodes for it, so a big set can't be starved by
        # higher-priority jobs that grab freed nodes first.
        overdue = [k for k, t in waiting_since.items() if time.time() - t > RESERVE_S
                   and stat[k]["pass"] < TARGET]
        # least-recently-served first (a config that just ran yields to one that never ran)
        head = min(overdue, key=lambda k: (last_launch.get(k, 0), waiting_since[k])) if overdue else None
        for key in CONFIGS:
            bench, nn, gbs, mbs, cp = key
            s = stat[key]
            inflight = sum(1 for ru in running.values() if ru["key"] == key)
            if s["pass"] + inflight >= TARGET:
                continue
            if s["pass"] == 0 and s["fail"] >= 2:
                continue                                    # blocked config
            if s["pass"] == 0 and inflight >= 1:
                continue                                    # gate run in flight
            if len(free) < nn:
                # only a config with nothing in flight is starving; one with a run going is progressing,
                # and reserving for it would idle nodes that smaller configs could use meanwhile
                if inflight == 0:
                    waiting_since.setdefault(key, time.time())
                else:
                    waiting_since.pop(key, None)
                continue
            if head is not None and key != head and len(free) - nn < head[1]:
                waiting_since.setdefault(key, time.time())
                continue                                    # would eat the head's reservation
            waiting_since.pop(key, None)
            nodes, free = free[:nn], free[nn:]
            slot_n += 1; slot = f"s{int(time.time()) % 100000}-{slot_n}"
            used = {int(r["seed"]) for r in load_runs() if r["seed"].isdigit()} | {ru["seed"] for ru in running.values()}
            seed = 1000 * nn + {"llama31_8b": 0, "llama2_70b_lora": 500, "flux1": 700}[bench] + 1
            while seed in used:
                seed += 1
            out = f"{OUT}/runlogs/{bench}-{nn}n-seed{seed}.out"
            log(f"LAUNCH {bench} {nn}n gbs={gbs} mbs={mbs} cp={cp} seed={seed} slot={slot} nodes={','.join(nodes)}")
            if DRY:
                continue
            label(nodes, slot)
            env = dict(os.environ, IMAGE=IMAGES[bench], BENCH=bench, NN=str(nn), GBS=str(gbs), MBS=str(mbs), CP=str(cp),
                       SEED=str(seed), TAG=f"v61-{sysname(nn)}", SELKEY=SELKEY, SELVAL=slot, PULL_SECRET="ccr",
                       TIMEOUT=str(TIMEOUT.get(bench, {}).get(nn, 2 * 3600)), N=str(nn))
            # run a private copy: bash reads scripts incrementally, so editing the source
            # while a run is in flight would corrupt it
            rdir, script, files, _ = RUNNERS[bench]
            snap = f"{OUT}/runlogs/.run-{slot}"; os.makedirs(snap, exist_ok=True)
            for f in files:
                sh(f"cp {rdir}/{f} {snap}/")
            args = [FLUX_SET[gbs], str(seed)] if bench == "flux1" else []
            p = subprocess.Popen(["bash", f"{snap}/{script}", *args], env=env, stdout=open(out, "w"), stderr=subprocess.STDOUT)
            running[slot] = dict(proc=p, key=key, nodes=nodes, seed=seed, out=out, start=time.time())
            last_launch[key] = time.time()
            launched_any = True
        if DRY:
            return
        todo = [key for key in CONFIGS if stat[key]["pass"] < TARGET and not (stat[key]["pass"] == 0 and stat[key]["fail"] >= 2)]
        if not running and not todo:
            log("ALL CONFIGS COMPLETE"); return
        if not running and not launched_any and todo and not free:
            log("ALERT no schedulable nodes")
        time.sleep(60)

if __name__ == "__main__":
    main()
