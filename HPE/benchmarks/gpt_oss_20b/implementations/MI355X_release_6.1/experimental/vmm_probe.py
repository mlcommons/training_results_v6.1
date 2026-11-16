#!/usr/bin/env python3
"""Probe HIP virtual-memory (VMM) peer access, the path RCCL uses with
NCCL_CUMEM_ENABLE=1 (and Megatron's rccl_sdma param all-gather).

RCCL fails on lodin with HIP 'invalid argument' in hipMemSetAccess, both in
allocator.cc (granting a peer GPU access to a local VMM buffer) and in p2p.cc
(setting access on a buffer imported from a peer process). This calls the HIP
API directly with ctypes, without RCCL or torch:

  1. hipMemCreate on GPU 0, map it, hipMemSetAccess for GPU 0    (local)
  2. same buffer, hipMemSetAccess for GPU 1..7                    (peer, single process)
  3. export as POSIX fd, import in a child process on GPU 1, map,
     hipMemSetAccess for GPU 1                                    (cross-process import)
  4. legacy path for comparison: hipDeviceCanAccessPeer / hipDeviceEnablePeerAccess
"""
import ctypes, glob, os, socket, sys, array

libs = sorted(glob.glob("/opt/venv/lib/python3.12/site-packages/_rocm_sdk_*/lib/libamdhip64.so*")) + \
       sorted(glob.glob("/opt/venv/lib/python3.12/site-packages/**/libamdhip64.so*", recursive=True)) + \
       sorted(glob.glob("/opt/rocm*/lib/libamdhip64.so*"))
hip = ctypes.CDLL(libs[0])
print("libamdhip64:", libs[0])

class Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]
class AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte), ("usage", ctypes.c_ushort)]
class Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleType", ctypes.c_int), ("location", Loc),
                ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", AllocFlags)]
class AccessDesc(ctypes.Structure):
    _fields_ = [("location", Loc), ("flags", ctypes.c_int)]

PINNED, POSIX_FD, LOC_DEVICE, PROT_RW, GRAN_RECOMMENDED = 1, 1, 1, 3, 1

def err(rc):
    hip.hipGetErrorString.restype = ctypes.c_char_p
    return "ok" if rc == 0 else f"rc={rc} {hip.hipGetErrorString(rc).decode()}"

def check(name, rc):
    print(f"  {name:55s} {err(rc)}"); sys.stdout.flush()
    return rc == 0

def prop_for(dev):
    p = Prop(); p.type = PINNED; p.requestedHandleType = POSIX_FD
    p.location = Loc(LOC_DEVICE, dev); return p

def access(dev):
    return AccessDesc(Loc(LOC_DEVICE, dev), PROT_RW)

n = ctypes.c_int(); hip.hipGetDeviceCount(ctypes.byref(n)); ndev = n.value
drv = ctypes.c_int(); rt = ctypes.c_int()
hip.hipDriverGetVersion(ctypes.byref(drv)); hip.hipRuntimeGetVersion(ctypes.byref(rt))
print(f"devices={ndev} hipDriverGetVersion={drv.value} hipRuntimeGetVersion={rt.value}")

attr = ctypes.c_int()
# hipDeviceAttributeVirtualMemoryManagementSupported
for name, code in (("VirtualMemoryManagementSupported", 10057),):
    rc = hip.hipDeviceGetAttribute(ctypes.byref(attr), code, 0)
    print(f"  attr {name}: {attr.value if rc == 0 else err(rc)}")

print("\n[4] legacy peer access")
for peer in range(1, ndev):
    c = ctypes.c_int(); hip.hipDeviceCanAccessPeer(ctypes.byref(c), 0, peer)
    print(f"  hipDeviceCanAccessPeer(0,{peer}) = {c.value}")

print("\n[1] VMM local")
hip.hipSetDevice(0)
p = prop_for(0); gran = ctypes.c_size_t()
check("hipMemGetAllocationGranularity", hip.hipMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(p), GRAN_RECOMMENDED))
size = max(gran.value, 2 << 20)
h = ctypes.c_ulonglong()
check(f"hipMemCreate(GPU0, {size} B, POSIX fd)", hip.hipMemCreate(ctypes.byref(h), ctypes.c_size_t(size), ctypes.byref(p), ctypes.c_ulonglong(0)))
ptr = ctypes.c_void_p()
check("hipMemAddressReserve", hip.hipMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_size_t(gran.value), None, ctypes.c_ulonglong(0)))
check("hipMemMap", hip.hipMemMap(ptr, ctypes.c_size_t(size), ctypes.c_size_t(0), h, ctypes.c_ulonglong(0)))
a = access(0)
check("hipMemSetAccess(GPU0)", hip.hipMemSetAccess(ptr, ctypes.c_size_t(size), ctypes.byref(a), ctypes.c_size_t(1)))

print("\n[2] VMM peer access, same process (RCCL allocator.cc:94)")
for peer in range(1, 2 if os.environ.get("PEER_ONLY") == "1" else ndev):
    a = access(peer)
    check(f"hipMemSetAccess(GPU0 buffer -> GPU{peer})", hip.hipMemSetAccess(ptr, ctypes.c_size_t(size), ctypes.byref(a), ctypes.c_size_t(1)))
a_all = (AccessDesc * ndev)(*[access(d) for d in range(ndev)])
check(f"hipMemSetAccess(GPU0 buffer -> all {ndev} GPUs at once)", hip.hipMemSetAccess(ptr, ctypes.c_size_t(size), a_all, ctypes.c_size_t(ndev)))

if os.environ.get("PEER_ONLY") == "1":
    print("\ndone (PEER_ONLY)"); sys.exit(0)
print("\n[3] VMM export -> import in another process (RCCL p2p.cc:374)")
fd = ctypes.c_int(-1)
ok = check("hipMemExportToShareableHandle(POSIX fd)", hip.hipMemExportToShareableHandle(ctypes.byref(fd), h, POSIX_FD, ctypes.c_ulonglong(0)))
if ok:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    pid = os.fork()
    if pid == 0:
        parent.close()
        msg, anc, _, _ = child.recvmsg(16, socket.CMSG_LEN(4))
        rfd = array.array("i", anc[0][2][:4])[0]
        hip2 = ctypes.CDLL(libs[0])
        globals()["hip"] = hip2
        for dev in (1, 0):
            print(f"  child: importing on GPU{dev}"); sys.stdout.flush()
            hip2.hipSetDevice(dev)
            ih = ctypes.c_ulonglong()
            if not check("  hipMemImportFromShareableHandle", hip2.hipMemImportFromShareableHandle(ctypes.byref(ih), ctypes.c_void_p(rfd), POSIX_FD)):
                continue
            iptr = ctypes.c_void_p()
            check("  hipMemAddressReserve", hip2.hipMemAddressReserve(ctypes.byref(iptr), ctypes.c_size_t(size), ctypes.c_size_t(0), None, ctypes.c_ulonglong(0)))
            check("  hipMemMap", hip2.hipMemMap(iptr, ctypes.c_size_t(size), ctypes.c_size_t(0), ih, ctypes.c_ulonglong(0)))
            ia = access(dev)
            check(f"  hipMemSetAccess(imported buffer -> GPU{dev})", hip2.hipMemSetAccess(iptr, ctypes.c_size_t(size), ctypes.byref(ia), ctypes.c_size_t(1)))
        os._exit(0)
    child.close()
    parent.sendmsg([b"x"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd.value]))])
    os.waitpid(pid, 0)
print("\ndone")
