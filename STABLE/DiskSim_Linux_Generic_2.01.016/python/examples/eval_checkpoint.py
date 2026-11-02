"""Evaluate a saved DQN checkpoint (greedy policy) against FCFS and SSTF.

  python3 -I examples/eval_checkpoint.py results/models/<tag>.pt [--seeds 1000 1010] [--interarrival 10]

Rebuilds the network and the environment settings stored in the checkpoint.
--interarrival overrides the mean request spacing (ms) to test a different load."""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, gymnasium as gym
import disksim_env
from baselines import fcfs, sstf, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--seeds", nargs=2, type=int, default=[1000, 1010], metavar=("START", "STOP"))
ap.add_argument("--interarrival", type=float, default=None, help="override mean interarrival (ms)")
ap.add_argument("--no-baselines", action="store_true")
ap.add_argument("--recycle-every", type=int, default=250, help="replace the DiskSim worker every N resets (0 = in-process)")
a = ap.parse_args()

ck = torch.load(a.checkpoint, weights_only=True)
h = ck["hidden"]
p = ck.get("train", {}).get("policy_dropout", 0.0)
if "3.weight" in ck["state_dict"]:        # layout with Dropout layers (Linear at indices 0, 3, 6)
    net = nn.Sequential(nn.Linear(ck["nin"], h), nn.ReLU(), nn.Dropout(p),
                        nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
                        nn.Linear(h, ck["nout"]))
else:                                      # older checkpoints without Dropout (Linear at 0, 2, 4)
    net = nn.Sequential(nn.Linear(ck["nin"], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ck["nout"]))
net.load_state_dict(ck["state_dict"]); net.eval()

kw = dict(ck["env_kwargs"])
if a.interarrival is not None:
    kw["mean_interarrival_ms"] = a.interarrival
env = gym.make("DiskSchedRecycled-v0", recycle_every=a.recycle_every, **kw) if a.recycle_every > 0 else gym.make("DiskSched-v0", **kw)
assert env.observation_space.shape[0] == ck["nin"] and env.action_space.n == ck["nout"], "env does not match checkpoint"

def greedy(obs, mask=None, rng=None):
    with torch.no_grad():
        q = net(torch.as_tensor(obs))
        if ck["use_mask"] and mask is not None:
            q = q.masked_fill(~torch.as_tensor(mask), -1e9)
        return int(q.argmax())

print(f"checkpoint: {ck['tag']}")
print(f"env: {kw}")
seeds = range(*a.seeds)
pols = [("DQN greedy", greedy)] if a.no_baselines else [("FCFS", fcfs), ("SSTF", sstf), ("DQN greedy", greedy)]
for n, p in pols:
    v = evaluate(p, env, seeds, return_all=True); m, sd = np.nanmean(v), np.nanstd(v)
    print(f"{n:11s} mean response {m:8.2f} ms (spread ±{sd:.2f}, std. error ±{sd / np.sqrt(len(v)):.2f}, n={len(v)})  seeds {seeds.start}-{seeds.stop-1}")
