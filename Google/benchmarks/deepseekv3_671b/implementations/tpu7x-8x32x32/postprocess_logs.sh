#!/usr/bin/env bash
# postprocess_logs.sh: build one MLPerf Training result file (result_*.txt) from the canonical
# MLLOG of a run plus the MaxText log lines of its exported Cloud Logging entries.
#
# training_rules.adoc "Result files": result logs must include the full framework logs and all
# MLLOG lines; internal paths and environment variables may be redacted; logs may be truncated
# when the file is too large.
#
# Usage:
#   gcloud logging read 'resource.type="k8s_container" AND
#       resource.labels.pod_name="<pod of JAX process 0>"' --order=asc --format=json > process0.json
#   postprocess_logs.sh --mllog mllog_<seed>.log --source-root <maxtext checkout>/src \
#       --log maxtext=process0.json --out result_0.txt \
#       [--platform <submission_platform>] [--runtime-prefix /app/src] [--max-mib 49] \
#       [--redact REGEX ...] [--redact-file FILE]
#
# --source-root DIR  MaxText source tree that ran (src/ of a checkout of the commit in ../maxtext/SOURCE_VERSION.txt,
#                   which the Dockerfile copies to /app/src).
#                   Selects the formatted/redacted framework log lines of JAX process 0 emitted from
#                   that tree (entries whose sourceLocation names a logging call in a .py file under
#                   DIR, or Python warnings starting with "<--runtime-prefix>/<path under DIR>.py:<line>:").
#                   Without --source-root every Cloud Logging entry of JAX process 0 is included.
# --platform        optional; rewrites the MLLOG submission_platform value.
#
# --log LABEL=FILE  FILE is JSON lines or a JSON array of Cloud Logging LogEntry objects
#                   ("timestamp" + "textPayload"/"jsonPayload", e.g. `gcloud logging read
#                   --format=json`), or text whose lines start with an RFC 3339 timestamp
#                   (e.g. `kubectl logs --timestamps`). Repeatable; LABEL becomes the "[LABEL] " tag.
#
# Output:
#   * framework lines as "[LABEL] <UTC timestamp> <text>", merged in timestamp order;
#   * the canonical MLLOG lines byte-for-byte (only the submission_platform value is rewritten),
#     in their original order, each placed at its time_ms position (running maximum, so the order
#     never changes): the training events land among the training logs and eval_start..eval_stop
#     among the eval logs. The last canonical line (run_stop, backdated time_ms) stays last;
#   * framework lines containing ":::MLLOG" are dropped: the checkers parse every ":::MLLOG"
#     occurrence, so the canonical file is the only source of MLLOG lines;
#   * REDACTIONS (below), then the site-specific patterns of --redact-file (one regex per line,
#     optionally followed by a tab and its replacement) and --redact, are applied to framework
#     lines only, never to MLLOG lines;
#   * above --max-mib, framework lines are cut from the middle (head and tail kept, one marker
#     line); MLLOG lines are never cut. Default 49 MiB: the mlperf_logging repo_checker rejects
#     files > 50 MiB (GitHub's hard limit, cited by the rules, is 100 MiB).
set -euo pipefail
exec python3 -I - "$@" <<'PYTHON'  # -I: ignore the working directory and PYTHONPATH
import argparse, calendar, json, re, sys, time

R = "<redacted>"
# ------------------------------------------------------------------------------------------------
# REDACTIONS: (regex, replacement), applied in this order to every framework line.
# ------------------------------------------------------------------------------------------------
SECRET_NAME = r"[A-Z][A-Z0-9_]*(?:_TOKEN|_KEY|_KEYS|SECRET[A-Z0-9_]*|PASSWORD|PASSWD|_CREDENTIALS?)"
SECRET_KEY = r"(?:[A-Za-z0-9]+_)*(?:api_key|apikey|access_token|auth_token|refresh_token|hf_token|hub_token|secret_key|client_secret|password|passwd)"
REDACTIONS = [
    # secrets
    (r"\bhf_[A-Za-z0-9]{20,}", R),                                              # Hugging Face tokens
    (r"\bgh[pousr]_[A-Za-z0-9]{20,}|\bAIza[0-9A-Za-z_-]{35}|\bya29\.[0-9A-Za-z_.-]+", R),  # GitHub/Google keys, OAuth
    (r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]{8,}", r"\1" + R),                     # Authorization: Bearer ...
    (r"(?i)(\bwandb\s+login\s+(?:--relogin\s+)?)[A-Za-z0-9_-]{20,}", r"\1" + R),  # wandb login <key>
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*", R),                               # PEM private keys
    (r"\b(" + SECRET_NAME + r")\b(['\"]?\s*[=:]\s*['\"]?)[^\s'\",;})\]]+", r"\1\2" + R),   # *_TOKEN=, *_KEY: ...
    (r"(?i)\b(" + SECRET_KEY + r")\b(['\"]?\s*[=:]\s*['\"]?)[^\s'\",;})\]]+", r"\1\2" + R),  # hf_token='...'
    # internal names
    (r"\b(?:[a-z0-9-]+\.)?(?:gcr\.io|pkg\.dev)/[a-z][a-z0-9-]*", "<registry>"),  # registry host + project
    (r"gs://[a-z0-9][a-z0-9._-]*", "gs://<bucket>"),                            # GCS buckets (+ bare names, below)
    (r"\b[a-z0-9-]+@[a-z0-9-]+\.iam\.gserviceaccount\.com|\b\d+-compute@developer\.gserviceaccount\.com", "<service-account>"),
    (r"\b[A-Za-z0-9._%+-]+@(?:google\.com|[a-z0-9.-]*\.google\.com)\b", "<user>@<domain>"),  # user emails
    (r"\bprojects/[a-z0-9][a-z0-9-]*", "projects/<project>"),                   # GCP project ids/numbers
    (r"\bgke-[a-z0-9-]+-[0-9a-f]{8}-[a-z0-9]{4}\b", "<node>"),                  # GKE node names
    (r"\b[a-z0-9.-]+\.(?:corp\.google\.com|googleplex\.com|prod\.google\.com|googlers\.com)\b", "<internal-host>"),
    (r"https?://wandb\.ai/\S+", "https://wandb.ai/<redacted>"),                 # private W&B project/run URLs
    (r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\.?\d)(?!\w)",                       # IPv4 (keeps 0.0.0.0, 127.0.0.1)
     lambda m: m.group(0) if m.group(0) in ("0.0.0.0", "127.0.0.1") else "<ip>"),
    (r"(?i)(?<![\w:])(?:(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}|[0-9a-f]{1,4}(?::[0-9a-f]{1,4})+::(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4})*)?"
     r"|[0-9a-f]{1,4}::[0-9a-f]{1,4}(?::[0-9a-f]{1,4})+)(?![\w:])", "<ip>"),    # IPv6 (not ::1)
]
# ------------------------------------------------------------------------------------------------

ap = argparse.ArgumentParser(prog="postprocess_logs.sh")
ap.add_argument("--mllog", required=True, help="canonical MLLOG file of the run")
ap.add_argument("--platform", help="submission_platform value to write (default: keep)")
ap.add_argument("--source-root", help="format/redact framework log lines of JAX process 0 from .py files under this tree")
ap.add_argument("--runtime-prefix", default="/app/src", help="path of --source-root at run time")
ap.add_argument("--log", action="append", default=[], metavar="LABEL=FILE")
ap.add_argument("--out", required=True)
ap.add_argument("--max-mib", type=float, default=49.0)
ap.add_argument("--redact", action="append", default=[], metavar="REGEX", help="extra pattern -> " + R)
ap.add_argument("--redact-file", action="append", default=[], metavar="FILE",
                help="site-specific patterns: one regex per line, optionally TAB replacement")
a = ap.parse_args()

TS = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:?\d\d)")
def ns(ts):  # RFC 3339 -> integer ns since epoch
    m = TS.fullmatch(ts)
    sec = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S"))
    if m.group(3) != "Z":
        sign, hh, mm = m.group(3)[0], int(m.group(3)[1:3]), int(m.group(3)[-2:])
        sec -= (1 if sign == "+" else -1) * (hh * 3600 + mm * 60)
    return sec * 10**9 + int(((m.group(2) or "") + "000000000")[:9])
def iso(n):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(n // 10**9)) + ".%06dZ" % (n % 10**9 // 1000)

# --source-root: basename -> paths of the .py files of the tree (sourceLocation has basenames only)
import os
SRC = {}
if a.source_root:
    for d, _, fs in os.walk(a.source_root):
        for f in fs:
            if f.endswith(".py"):
                SRC.setdefault(f, []).append(os.path.join(d, f))
_lines = {}
def src_line(p, n):
    if p not in _lines:
        _lines[p] = open(p, encoding="utf-8", errors="replace").read().split("\n")
    return _lines[p][n - 1] if 0 < n <= len(_lines[p]) else ""
LOGCALL = re.compile(r"\b(?:max_logging\.log|log(?:ger|ging)?\.(?:debug|info|warning|warn|error|exception|critical|log))\s*\(")
WARN = re.compile(re.escape(a.runtime_prefix.rstrip("/")) + r"/(\S+?\.py):\d+: ")
def from_source(e, t):  # Cloud Logging entry emitted by code under --source-root?
    sl = e.get("sourceLocation") or {}
    f, n = sl.get("file", ""), int(sl.get("line") or 0)
    if f.endswith(".py") and any(LOGCALL.search(src_line(p, n)) for p in SRC.get(os.path.basename(f), [])):
        return True
    m = WARN.match(t)
    return bool(m) and os.path.isfile(os.path.join(a.source_root, m.group(1)))

kept, skipped = 0, 0
def read_log(path):  # yields (ns, text)
    global kept, skipped
    raw = open(path, encoding="utf-8", errors="replace").read()
    head = raw.lstrip()[:1]
    if head in ("[", "{"):
        es = json.loads(raw) if head == "[" else (json.loads(l) for l in raw.splitlines() if l.strip())
        for e in es:
            p = e.get("jsonPayload")
            t = e["textPayload"] if "textPayload" in e else (
                p["message"] if isinstance(p, dict) and set(p) <= {"message", "pid"} and "message" in p
                else json.dumps(p, ensure_ascii=False))
            if a.source_root and not from_source(e, t):
                skipped += 1; continue
            kept += 1
            yield ns(e["timestamp"]), t
    else:
        assert not a.source_root, "--source-root needs Cloud Logging JSON entries (sourceLocation)"
        last = None
        for l in raw.splitlines():
            m = re.match(r"(\S+)[ \t](.*)", l)
            if m and TS.fullmatch(m.group(1)):
                last = ns(m.group(1)); yield last, m.group(2)
            elif last is not None:
                yield last, l  # continuation line

# canonical MLLOG (only the submission_platform value changes)
mllog = [l.rstrip("\r\n") for l in open(a.mllog, encoding="utf-8") if l.strip()]
assert all(l.startswith(":::MLLOG ") for l in mllog), "canonical file has non-MLLOG lines"
recs = [json.loads(l[len(":::MLLOG "):]) for l in mllog]
assert recs[-1]["key"] == "run_stop", "last canonical line must be run_stop"
for i, r in enumerate(recs):
    if r["key"] == "submission_platform" and a.platform:
        old, new = '"value": ' + json.dumps(r["value"]), '"value": ' + json.dumps(a.platform)
        assert mllog[i].count(old) == 1
        mllog[i] = mllog[i].replace(old, new)

# framework lines
fw, dropped, buckets, users = [], 0, set(), set()
for k, spec in enumerate(a.log):
    label, path = spec.split("=", 1)
    for seq, (t, text) in enumerate(read_log(path)):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        for line in (text[:-1] if text.endswith("\n") else text).split("\n"):
            if ":::MLLOG" in line:
                dropped += 1; continue
            buckets.update(re.findall(r"gs://([a-z0-9][a-z0-9._-]{5,})", line))
            users.update(re.findall(r"\b([a-z][a-z0-9_.-]{2,})@google\.com\b", line))
            fw.append((t, k, seq, label, line))
fw.sort(key=lambda x: x[:3])

def alt(words):
    return r"(?<![A-Za-z0-9])(?:" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")(?![A-Za-z0-9])"
rules = list(REDACTIONS)
i = rules.index(next(r for r in rules if r[1] == "gs://<bucket>")) + 1
rules[i:i] = [(alt(buckets), "<bucket>")] if buckets else []
rules += [(alt(users), "<user>")] if users else []
for path in a.redact_file:
    for l in open(path, encoding="utf-8"):
        l = l.rstrip("\n")
        if l.strip() and not l.lstrip().startswith("#"):
            p, _, rep = l.partition("\t")
            rules.append((p, rep or R))
rules += [(x, R) for x in a.redact]
rules = [(re.compile(p, re.S), rep) for p, rep in rules]
def redact(s):
    for rx, rep in rules:
        s = rx.sub(rep, s)
    return s

# merge: MLLOG line i goes right before the first framework line later than max(time_ms[:i+1])
out, j, anchor = [], 0, 0
for line, r in zip(mllog[:-1], recs[:-1]):
    anchor = max(anchor, int(r["time_ms"]) * 10**6)
    while j < len(fw) and fw[j][0] < anchor:
        out.append((1, fw[j])); j += 1
    out.append((0, line))
out += [(1, x) for x in fw[j:]] + [(0, mllog[-1])]
lines = [s if kind == 0 else "[%s] %s %s" % (s[3], iso(s[0]), redact(s[4])) for kind, s in out]

# truncate framework lines from the middle if needed (MLLOG lines are kept)
limit = int(a.max_mib * 2**20)
size = sum(len(l.encode()) + 1 for l in lines)
if size > limit:
    fwi = [n for n, (kind, _) in enumerate(out) if kind == 1]
    budget = (limit - sum(len(lines[n].encode()) + 1 for n, (k, _) in enumerate(out) if k == 0) - 512) // 2
    head = tail = 0; h = 0; t = len(fwi)
    while h < t and head + len(lines[fwi[h]].encode()) + 1 <= budget:
        head += len(lines[fwi[h]].encode()) + 1; h += 1
    while t > h and tail + len(lines[fwi[t - 1]].encode()) + 1 <= budget:
        tail += len(lines[fwi[t - 1]].encode()) + 1; t -= 1
    cut = set(fwi[h:t])
    assert cut, "MLLOG lines alone exceed --max-mib"
    nbytes = sum(len(lines[n].encode()) + 1 for n in cut)
    marker = "[postprocess] %d framework lines (%d bytes, %s .. %s) truncated to keep this file under %g MiB" % (
        len(cut), nbytes, iso(out[fwi[h]][1][0]), iso(out[fwi[t - 1]][1][0]), a.max_mib)
    lines = [marker if n == fwi[h] else l for n, l in enumerate(lines) if n not in cut or n == fwi[h]]

assert [l for l in lines if ":::MLLOG" in l] == mllog, "MLLOG lines changed"  # exactly the canonical lines, in order
with open(a.out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
nml = sum(1 for l in lines if ":::MLLOG" in l)
print("%s: %d lines (%d MLLOG, %d framework), %d bytes; dropped %d framework :::MLLOG lines; %s" % (
    a.out, len(lines), nml, len(lines) - nml, sum(len(l.encode()) + 1 for l in lines), dropped,
    "TRUNCATED" if size > limit else "not truncated"), file=sys.stderr)
if a.source_root:
    print("%s: kept %d log entries emitted under --source-root, skipped %d" % (a.out, kept, skipped), file=sys.stderr)
PYTHON
