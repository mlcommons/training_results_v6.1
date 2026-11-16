#!/bin/bash
# lustre_mount_guard.sh — fail fast if the training data path is not a real Lustre mount.
#
# WHY: /lustre/agent is a SYMLINK to /mlstorage/lustre_agent_work, and the Lustre client
# is mounted ON TOP of that ext4 directory. If the client mount is missing (e.g. after a
# reboot where lustre-lifecycle.service failed), every path still RESOLVES and every write
# SUCCEEDS -- straight onto the ext4 root filesystem. A 9-day 10-run chain would silently
# produce garbage and could fill /mlstorage. This guard makes that impossible.
#
# Usage:  lustre_mount_guard.sh [path]       (default: /lustre/agent)
# Exit:   0 = verified Lustre mount, non-zero = NOT safe to run
#
# Call this from run.sub / submit_*.sh BEFORE any training work starts.

set -uo pipefail

TARGET="${1:-/lustre/agent}"
RC=0

die() { echo "[mount-guard] FATAL: $*" >&2; RC=1; }
ok()  { echo "[mount-guard] OK: $*"; }

# --- 1. resolve the symlink to its real path --------------------------------------
if [ ! -e "$TARGET" ]; then
    die "$TARGET does not exist"
    exit 1
fi
REAL=$(readlink -f "$TARGET" 2>/dev/null)
if [ -z "$REAL" ]; then
    die "cannot resolve $TARGET"
    exit 1
fi
[ "$REAL" != "$TARGET" ] && ok "$TARGET -> $REAL (symlink resolved)"

# --- 2. filesystem type must be lustre (statfs magic) -----------------------------
# stat -f -c %T prints the fs type name; for Lustre it is "lustre".
FSTYPE=$(stat -f -c %T "$REAL" 2>/dev/null)
if [ "$FSTYPE" != "lustre" ]; then
    die "$REAL is on filesystem type '${FSTYPE:-unknown}', expected 'lustre'."
    die "Lustre client is NOT mounted. Writes here would land on the local disk."
fi
[ $RC -eq 0 ] && ok "statfs type = lustre"

# --- 3. it must be an actual mountpoint of type lustre ----------------------------
# (guards against the resolved path merely being *under* a lustre mount elsewhere)
if ! findmnt -t lustre --target "$REAL" >/dev/null 2>&1; then
    die "findmnt -t lustre --target $REAL found no lustre mount"
fi
if [ $RC -eq 0 ]; then
    SRC=$(findmnt -n -o SOURCE --target "$REAL" 2>/dev/null)
    MNT=$(findmnt -n -o TARGET --target "$REAL" 2>/dev/null)
    ok "mount: $SRC on $MNT"
fi

# --- 4. Lustre servers must be reachable ------------------------------------------
if command -v lfs >/dev/null 2>&1; then
    if ! lfs df "$REAL" >/dev/null 2>&1; then
        die "lfs df failed on $REAL -- Lustre targets not reachable"
    else
        INACTIVE=$(lfs check servers 2>/dev/null | grep -vc 'active' || true)
        if [ "${INACTIVE:-0}" -gt 0 ]; then
            die "lfs check servers reports $INACTIVE non-active target(s)"
            lfs check servers 2>&1 | grep -v 'active' >&2
        else
            ok "lfs df / lfs check servers healthy"
        fi
    fi
fi

# --- 5. must be writable by the job user ------------------------------------------
PROBE="${REAL}/.mount_guard_$$"
if ! ( umask 0002; : > "$PROBE" ) 2>/dev/null; then
    die "cannot create a file under $REAL (permissions or read-only mount)"
else
    rm -f "$PROBE" 2>/dev/null
    ok "write probe succeeded"
fi

if [ $RC -ne 0 ]; then
    echo "[mount-guard] ABORTING: refusing to start training on a non-Lustre path." >&2
    echo "[mount-guard] Check:  systemctl status lustre-lifecycle.service" >&2
    echo "[mount-guard]         journalctl -u lustre-lifecycle.service -b" >&2
    echo "[mount-guard] Repair: sudo systemctl restart lustre-lifecycle.service" >&2
    exit 1
fi

echo "[mount-guard] PASS: $TARGET is a verified Lustre mount."
exit 0
