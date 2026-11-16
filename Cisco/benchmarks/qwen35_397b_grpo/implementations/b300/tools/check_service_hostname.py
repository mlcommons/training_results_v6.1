''                                                                             
import ipaddress
import socket
import sys

expected = sys.argv[1]
hostname = socket.gethostname()
resolved = socket.gethostbyname(hostname)
address = ipaddress.ip_address(resolved)
if address.is_loopback or address.is_unspecified or resolved != expected:
    raise SystemExit(
        f"ERROR: {hostname} resolves to {resolved}, expected {expected}; "
        "Gym would bind/advertise an unreachable address. Fix hostname mapping before launch."
    )
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.bind((resolved, 0))
print(f"GYM_HOSTNAME_CHECK: PASS host={hostname} address={resolved}", flush=True)
