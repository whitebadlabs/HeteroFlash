import socket, sys, time, statistics
mode, host, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
def recvn(c, n):
    b = bytearray()
    while len(b) < n:
        d = c.recv(n - len(b))
        if not d: raise EOFError
        b += d
    return b
if mode == "srv":
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((host, port)); s.listen(1); s.settimeout(60)
    c, _ = s.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            n = int.from_bytes(recvn(c, 4), "little"); recvn(c, n); c.sendall(n.to_bytes(4, "little") + b"\0" * n)
    except EOFError: pass
else:
    c = socket.create_connection((host, port)); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for n in (64, 8192, 16384, 65536):
        ts = []
        for i in range(400):
            t = time.perf_counter(); c.sendall(n.to_bytes(4, "little") + b"\0" * n); recvn(c, 4 + n); ts.append(time.perf_counter() - t)
        ts = sorted(ts[50:]); print(f"{n:6d} B each way: median {statistics.median(ts)*1e3:.3f} ms, p90 {ts[int(len(ts)*.9)]*1e3:.3f} ms")
    c.close()
