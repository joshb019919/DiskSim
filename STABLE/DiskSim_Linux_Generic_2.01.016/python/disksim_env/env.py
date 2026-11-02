"""Gymnasium environment: pick which pending request the disk services next.

Python owns the queue; DiskSim only ever sees one request at a time, so the
agent (not DiskSim's internal scheduler) decides the order.

Action  : Discrete(queue_slots). Slot i = i-th oldest pending request.
Obs     : Box(float32), [head_cyl, head_angle, backlog] + queue_slots * 6 features
          per slot: valid, signed cyl delta, |cyl delta|, wait, is_read, size.
Reward  : see `reward_mode`, divided by `reward_scale`.
          Choosing an empty slot costs `invalid_penalty`, takes no simulated time, and
          after `max_consecutive_invalid` in a row the episode is truncated.
Episode : ends (truncated=True) after `episode_requests` completions, or early (with
          info["disksim_hang"]=True, reward 0) if DiskSim hits its event limit.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .dsim import DiskSim, DiskSimHang

HERE = os.path.dirname(__file__)
# Override with DISKSIM_PARFILE (a .parv whose model files sit beside it).
DEFAULT_PARFILE = os.path.abspath(os.environ.get("DISKSIM_PARFILE") or
                                  os.path.join(HERE, "..", "..", "valid", "cheetah4LP.parv"))
F = 6  # features per slot


class DiskSchedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, parfile=DEFAULT_PARFILE, lib_path=None, queue_slots=16,
                 episode_requests=500, mean_interarrival_ms=12.0, read_fraction=0.67,
                 sizes=(8, 16, 32), reward_mode="queue_time", reward_scale=100.0,
                 invalid_penalty=5.0, max_consecutive_invalid=10,
                 wait_scale_ms=200.0, outfile=os.devnull):
        assert reward_mode in ("response", "service", "queue_time")
        self.parfile, self.outfile = parfile, outfile
        self.K, self.N = queue_slots, episode_requests
        self.mean_ia, self.read_frac, self.sizes = mean_interarrival_ms, read_fraction, tuple(sizes)
        self.reward_mode, self.reward_scale = reward_mode, reward_scale
        self.invalid_penalty, self.wait_scale = invalid_penalty, wait_scale_ms
        self.max_invalid = max_consecutive_invalid
        self.sim = DiskSim(lib_path) if lib_path else DiskSim()
        self.sim.start(parfile, outfile)
        self.ncyl, _, self.nblocks = self.sim.geometry()
        self.sim.shutdown()
        self.action_space = spaces.Discrete(self.K)
        self.observation_space = spaces.Box(-1.0, 1.0, (3 + self.K * F,), np.float32)

    # ---- workload -------------------------------------------------------
    def _new_arrival(self):
        self._next_t += self.np_random.exponential(self.mean_ia)
        size = int(self.np_random.choice(self.sizes))
        lbn = int(self.np_random.integers(0, self.nblocks - size))
        cyl = self.sim.lbn_to_pbn(lbn)[0]
        self._next_req = dict(t=self._next_t, lbn=lbn, size=size, cyl=cyl,
                              read=bool(self.np_random.random() < self.read_frac))

    def _pull_arrivals(self, upto):
        """Move arrivals with t <= upto into the backlog; return them."""
        got = []
        while self._next_req["t"] <= upto:
            got.append(self._next_req)
            self.queue.append(self._next_req)
            self._new_arrival()
        return got

    # ---- gym API --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.shutdown()
        self.sim.start(self.parfile, self.outfile)
        self.T, self._next_t = 0.0, 0.0
        self.queue, self.done_n, self.steps, self.bad_run = [], 0, 0, 0
        self.hangs = getattr(self, "hangs", 0)   # cumulative over the env's lifetime
        self.sum_resp = 0.0
        self.head_cyl, self.head_angle = 0, 0.0
        self._new_arrival()
        self._pull_arrivals(self.T)
        self._ensure_work()
        return self._obs(), self._info()

    def _ensure_work(self):
        if not self.queue:  # idle: jump to next arrival
            self.T = self._next_req["t"]
            self._pull_arrivals(self.T)

    def action_masks(self):
        m = np.zeros(self.K, dtype=bool)
        m[:min(self.K, len(self.queue))] = True
        return m

    def _obs(self):
        o = np.zeros(3 + self.K * F, np.float32)
        o[0], o[1] = self.head_cyl / self.ncyl, self.head_angle
        o[2] = min(len(self.queue) / (4 * self.K), 1.0)
        for i, r in enumerate(self.queue[:self.K]):
            d = (r["cyl"] - self.head_cyl) / self.ncyl
            o[3 + i * F: 3 + (i + 1) * F] = (
                1.0, d, abs(d), min((self.T - r["t"]) / self.wait_scale, 1.0),
                float(r["read"]), r["size"] / max(self.sizes))
        return o

    def _info(self, **kw):
        return dict(action_mask=self.action_masks(), time_ms=self.T, queue_len=len(self.queue),
                    completed=self.done_n,
                    mean_response_ms=self.sum_resp / self.done_n if self.done_n else 0.0, **kw)

    def step(self, action):
        if not self.sim.open:
            raise RuntimeError("DiskSim is not open: call reset() first. Only one "
                               "DiskSchedEnv can be live per process (another env's "
                               "reset() takes over); use SubprocVecEnv for parallel envs.")
        self.steps += 1
        a = int(action)
        if a >= min(self.K, len(self.queue)):  # empty slot: no time passes
            self.bad_run += 1
            capped = self.bad_run >= self.max_invalid   # policy is stuck: end the episode
            trunc = capped or self.steps > 20 * self.N
            return self._obs(), -self.invalid_penalty, False, trunc, \
                self._info(invalid=True, invalid_cap=capped)
        self.bad_run = 0

        req = self.queue.pop(a)
        n0 = len(self.queue) + 1                      # in system at dispatch, incl. this one
        t0 = self.T
        try:
            self.sim.advance_to(t0)
            c = self.sim.service(req["lbn"], req["size"], req["read"])
        except DiskSimHang:
            # DiskSim spun forever (known bug); its state is corrupt, so end the episode here.
            self.hangs += 1
            self.sim.shutdown()
            return self._obs(), 0.0, False, True, self._info(disksim_hang=True, hangs=self.hangs)
        arrived = self._pull_arrivals(c)
        service, response = c - t0, c - req["t"]
        self.T = c
        self.done_n += 1
        self.sum_resp += response
        hs = self.sim.head_state()
        if hs:
            self.head_cyl, _, self.head_angle = hs
        if self.reward_mode == "response":
            r = -response
        elif self.reward_mode == "service":
            r = -service
        else:  # time-integral of requests in system: sums to total response time
            r = -(n0 * service + sum(c - x["t"] for x in arrived))
        truncated = self.done_n >= self.N
        if not truncated:
            self._ensure_work()
        return self._obs(), r / self.reward_scale, False, truncated, \
            self._info(response_ms=response, service_ms=service)

    def close(self):
        self.sim.shutdown()
