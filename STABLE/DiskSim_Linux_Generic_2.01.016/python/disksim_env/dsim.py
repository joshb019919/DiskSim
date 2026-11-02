"""Thin ctypes binding for src/libdsim.so (one DiskSim instance per process)."""
import ctypes as C
import os
from contextlib import contextmanager

# Override with DISKSIM_LIB; the relative default works from <root>/python or <root>/gym.
DEFAULT_LIB = os.environ.get("DISKSIM_LIB") or os.path.join(os.path.dirname(__file__), "..", "..", "src", "libdsim.so")


class Obs(C.Structure):
    _fields_ = [("cylinder", C.c_int), ("head", C.c_int), ("angle_turns", C.c_double)]


@contextmanager
def _chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


_TRACE = open(os.environ["DSIM_TRACE"], "w", buffering=1) if os.environ.get("DSIM_TRACE") else None


class DiskSimHang(RuntimeError):
    """DiskSim exceeded its event limit (it spins forever on some requests, e.g. after a
    track-crossing access). Its state is unusable: shut down and start() again."""


class DiskSim:
    _live = None  # DiskSim keeps global state: only one instance may be open.

    def __init__(self, lib_path=DEFAULT_LIB):
        self.lib = L = C.CDLL(os.path.abspath(lib_path))
        L.dsim_init.argtypes = [C.c_char_p, C.c_char_p]
        L.dsim_submit.argtypes = [C.c_int, C.c_ulong, C.c_int, C.c_int]
        L.dsim_run_until_done.restype = C.c_double
        L.dsim_drain.restype = C.c_double
        L.dsim_now.restype = C.c_double
        L.dsim_advance_to.argtypes = [C.c_double]
        L.dsim_advance_to.restype = C.c_int
        L.dsim_set_event_limit.argtypes = [C.c_long]
        L.dsim_get_pos.argtypes = [C.c_int, C.POINTER(Obs)]
        L.dsim_geometry.argtypes = [C.c_int] + [C.POINTER(C.c_int)] * 3
        L.dsim_lbn_to_pbn.argtypes = [C.c_int, C.c_int] + [C.POINTER(C.c_int)] * 3
        self.open = False

    def start(self, parfile, outfile=os.devnull):
        if DiskSim._live is not None:   # library state is global: last starter wins
            DiskSim._live.shutdown()
        parfile = os.path.abspath(parfile)
        # .parv files reference models by relative path.
        with _chdir(os.path.dirname(parfile)):
            rc = self.lib.dsim_init(os.path.basename(parfile).encode(), outfile.encode())
        if rc != 0:
            raise RuntimeError(f"dsim_init failed for {parfile}")
        self.open = True
        DiskSim._live = self

    def shutdown(self):
        if self.open:
            self.lib.dsim_shutdown()
            self.open = False
            DiskSim._live = None

    def geometry(self, devno=0):
        a, b, c = C.c_int(), C.c_int(), C.c_int()
        if self.lib.dsim_geometry(devno, C.byref(a), C.byref(b), C.byref(c)):
            raise RuntimeError("bad devno")
        return a.value, b.value, c.value  # cylinders, surfaces, total sectors

    def lbn_to_pbn(self, lbn, devno=0):
        a, b, c = C.c_int(), C.c_int(), C.c_int()
        if self.lib.dsim_lbn_to_pbn(devno, lbn, C.byref(a), C.byref(b), C.byref(c)):
            raise ValueError(f"lbn {lbn} does not exist")
        return a.value, b.value, c.value

    def now(self):
        return self.lib.dsim_now()

    def set_event_limit(self, n):
        self.lib.dsim_set_event_limit(n)

    def advance_to(self, t):
        if self.lib.dsim_advance_to(t):
            raise DiskSimHang("event limit hit while advancing time")

    def service(self, blkno, sectors, is_read, devno=0):
        """Submit one request at the current time; return its completion time (ms)."""
        if _TRACE:
            _TRACE.write(f"{self.now():.6f} {blkno} {sectors} {int(is_read)}\n")
        self.lib.dsim_submit(devno, blkno, sectors, int(is_read))
        t = self.lib.dsim_run_until_done()
        if t == -2.0:
            raise DiskSimHang(f"event limit hit servicing blk={blkno} sectors={sectors} read={is_read}")
        if t < 0:
            raise RuntimeError("DiskSim event queue ran dry before completion")
        return t

    def head_state(self, devno=0):
        o = Obs()
        if self.lib.dsim_get_pos(devno, C.byref(o)):
            return None
        return o.cylinder, o.head, o.angle_turns
