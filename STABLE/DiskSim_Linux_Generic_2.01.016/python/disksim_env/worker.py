"""Worker process for RecyclingEnv: owns ONE real DiskSchedEnv (and so one DiskSim) and serves requests
over a pipe. Run as:  python -I worker.py <read_fd> <write_fd>   (the parent passes both fds).
It exits when the parent closes its end, so a crashed parent never leaves it behind."""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <root>/python
from multiprocessing.connection import Connection

ATTRS = ("reward_mode", "reward_scale", "K", "N", "mean_ia", "read_frac", "sizes", "invalid_penalty",
         "max_invalid", "wait_scale", "ncyl", "nblocks")

def main():
    rd, wr = Connection(int(sys.argv[1]), writable=False), Connection(int(sys.argv[2]), readable=False)
    env = None
    while True:
        try:
            msg = rd.recv()
        except (EOFError, OSError):
            break                                              # parent is gone
        cmd = msg[0]
        try:
            if cmd == "make":
                from disksim_env.env import DiskSchedEnv
                env = DiskSchedEnv(**msg[1])
                wr.send(("ok", dict(observation_space=env.observation_space, action_space=env.action_space,
                                    attrs={a: getattr(env, a) for a in ATTRS})))
            elif cmd == "reset":
                wr.send(("ok", env.reset(seed=msg[1], options=msg[2])))
            elif cmd == "step":
                wr.send(("ok", env.step(msg[1])))
            elif cmd == "masks":
                wr.send(("ok", env.action_masks()))
            elif cmd == "close":
                if env is not None: env.close()
                wr.send(("ok", None)); break
        except Exception:
            wr.send(("err", traceback.format_exc()))
    os._exit(0)          # skip interpreter teardown: faster, and avoids touching DiskSim's global state again

if __name__ == "__main__":
    main()
