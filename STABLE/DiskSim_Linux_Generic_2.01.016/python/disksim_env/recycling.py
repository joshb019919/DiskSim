"""RecyclingEnv: a Gymnasium environment that runs DiskSchedEnv in a worker subprocess and replaces the
worker every `recycle_every` resets.

Why: DiskSim leaks ~0.5 MB every time it is (re)initialized (reset() does that), and never frees it. In a long
training run, with evaluation episodes, that adds up to gigabytes. Ending the worker process returns all of it to
the OS, so memory stays flat no matter how long the run is.

  env = gym.make("DiskSchedRecycled-v0", recycle_every=250, queue_slots=16, ...)   # kwargs go to DiskSchedEnv

Behaves like DiskSchedEnv (same spaces, same obs/reward/info, same results for the same seeds). If the worker
dies or stops answering, step() returns a truncated transition with info["worker_crashed"]=True and the next
reset() starts a fresh worker, so one bad episode never kills a training run.
Always pass seed= to reset(): a restarted worker cannot continue the old worker's random stream.
"""
import atexit, os, subprocess, sys
import numpy as np
import gymnasium as gym
from multiprocessing.connection import Connection

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
TIMEOUT_S = 600          # a worker silent this long is treated as dead


class RecyclingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, recycle_every=250, **env_kwargs):
        self.recycle_every, self._kw = int(recycle_every), env_kwargs
        self._proc = self._tx = self._rx = None
        self._resets = 0
        self.restarts = 0                      # how many workers have been started (for logging/tests)
        self._last_obs = None
        self._attrs = {}
        atexit.register(self._stop)
        self._start()
        # first start tells us the spaces
        self.restarts = 0

    # ---- worker management ------------------------------------------------
    def _start(self):
        p2c_r, p2c_w = os.pipe(); c2p_r, c2p_w = os.pipe()          # parent->child, child->parent
        self._proc = subprocess.Popen([sys.executable, "-I", WORKER, str(p2c_r), str(c2p_w)],
                                      pass_fds=(p2c_r, c2p_w), cwd=os.path.dirname(WORKER))
        os.close(p2c_r); os.close(c2p_w)                            # keep only our ends
        self._tx, self._rx = Connection(p2c_w, readable=False), Connection(c2p_r, writable=False)
        info = self._call(("make", self._kw))
        self.observation_space, self.action_space, self._attrs = info["observation_space"], info["action_space"], info["attrs"]
        self._resets = 0
        self.restarts += 1

    def _stop(self):
        p, self._proc = self._proc, None
        if p is None: return
        try:
            self._tx.send(("close",)); self._rx.poll(5)
        except Exception:
            pass
        for c in (self._tx, self._rx):
            try: c.close()
            except Exception: pass
        try: p.wait(timeout=5)
        except Exception:
            p.kill(); p.wait()

    def _call(self, msg):
        self._tx.send(msg)
        if not self._rx.poll(TIMEOUT_S):
            raise TimeoutError("DiskSim worker did not answer")
        status, payload = self._rx.recv()
        if status == "err":
            raise RuntimeError("DiskSim worker raised:\n" + payload)
        return payload

    def _alive(self):
        return self._proc is not None and self._proc.poll() is None

    # ---- gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self._alive() or self._resets >= self.recycle_every:
            self._stop(); self._start()                              # fresh process: leaked memory is released
        try:
            out = self._call(("reset", seed, options))
        except (EOFError, OSError, TimeoutError, BrokenPipeError):  # died during reset: one retry on a new worker
            self._stop(); self._start()
            out = self._call(("reset", seed, options))
        self._resets += 1
        self._last_obs = out[0]
        return out

    def step(self, action):
        try:
            out = self._call(("step", int(action)))
        except (EOFError, OSError, TimeoutError, BrokenPipeError):
            self._stop()                                             # next reset() starts a new worker
            obs = np.zeros_like(self._last_obs) if self._last_obs is not None else np.zeros(self.observation_space.shape, np.float32)
            return obs, 0.0, False, True, dict(worker_crashed=True, disksim_hang=False, completed=0, mean_response_ms=0.0,
                                               action_mask=np.zeros(self.action_space.n, bool))
        self._last_obs = out[0]
        return out

    def action_masks(self):
        return self._call(("masks",))

    def close(self):
        self._stop()

    def __getattr__(self, name):           # K, N, reward_mode, ... as on DiskSchedEnv (used by the training scripts)
        if name != "_attrs" and name in self.__dict__.get("_attrs", {}):
            return self._attrs[name]
        raise AttributeError(name)

    def __del__(self):
        try: self._stop()
        except Exception: pass
