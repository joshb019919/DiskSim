"""FCFS / SSTF / random baselines. Sanity check: SSTF should beat FCFS under load."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, gymnasium as gym
import disksim_env  # registers DiskSched-v0

F = 6

def fcfs(obs, mask, rng): return 0
def sstf(obs, mask, rng):
    d = obs[3:].reshape(-1, F)[:, 2].copy(); d[~mask] = 9; return int(d.argmin())
def rand(obs, mask, rng): return int(rng.choice(np.flatnonzero(mask)))

def evaluate(policy, env, seeds, return_all=False):
    """Mean and std of per-episode mean response over `seeds`; with return_all=True, the per-seed array instead."""
    out, rng, hung = [], np.random.default_rng(0), 0
    for s in seeds:
        obs, info = env.reset(seed=s); done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs, info["action_mask"], rng))
            done = term or trunc
        hung += bool(info.get("disksim_hang"))
        out.append(info["mean_response_ms"] if info["completed"] else np.nan)
    if hung:
        print(f"  note: {hung}/{len(out)} eval episodes ended early on a DiskSim hang (episode shorter than {env.unwrapped.N} requests)")
    if return_all:
        return np.array(out)
    bad = int(np.isnan(out).sum())
    if bad:
        print(f"  warning: {bad}/{len(out)} episodes completed no requests (policy stuck on invalid actions)")
    return (np.nanmean(out), np.nanstd(out)) if bad < len(out) else (float("nan"), float("nan"))

if __name__ == "__main__":
    ia = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    env = gym.make("DiskSched-v0", mean_interarrival_ms=ia)
    for name, p in [("FCFS", fcfs), ("SSTF", sstf), ("random", rand)]:
        m, s = evaluate(p, env, range(10))
        print(f"{name:7s} mean response {m:8.2f} ms (±{s:.2f} across seeds)  [interarrival {ia} ms]")
