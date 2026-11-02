import argparse
import math
import numpy as np
import os
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIMIZERS = {
    "adam":    torch.optim.Adam,
    "adamw":   torch.optim.AdamW,
    "sgd":     torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "nadam":   torch.optim.NAdam,
}
NEG = -1e9
HIST = []                                             
EVAL_HIST = []                                       # eval_hist: (episodes_done, greedy mean ms)
LAST = {}                                            # LAST: last snapshot's results

def atomic(write, path): 
    write(path + ".tmp"); os.replace(path + ".tmp", path)           # a crash mid-save never corrupts the last file


def greedy(obs, net, use_mask, mask=None, rng=None):
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            q = net(torch.as_tensor(obs))
            if use_mask and mask is not None:
                q = q.masked_fill(~torch.as_tensor(mask), NEG)
            return int(q.argmax())
    finally:
        net.train(was_training)


def last_train_mean(k=25):
    r = [x[3] for x in HIST[-k:] if x[4]]
    return float(np.mean(r)) if r else float("nan"), min(k, len(HIST))


def ms(v): 
    return (float(np.nanmean(v)), float(np.nanstd(v)))


def optimizer(name):
    try:
        return OPTIMIZERS[name.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"unknown optimizer {name!r}; choose from {', '.join(OPTIMIZERS)}")


def save_png(ep_done, res, n_eval, episodes, run, run_dir, seeds):
    h = np.array(HIST, dtype=float)
    ep_ = h[:, 0]
    eps_ = h[:, 1]
    lr_ = h[:, 2]
    resp = h[:, 3]

    w = max(1, min(25, len(h) // 10))
    ma = np.convolve(resp, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(ep_, resp, color="#9ecae1", lw=0.8, label="training, per episode")
    ax[0].plot(ep_[w - 1:], ma, color="#08519c", lw=2, label=f"training, {w}-episode mean")

    if EVAL_HIST:
        ee = np.array(EVAL_HIST)
        ax[0].plot(ee[:, 0] - 1, ee[:, 1], "o-", color="#ff7f0e", lw=1.5, label="greedy eval at snapshots")

    ax[0].axhline(res["SSTF"][0], color="#2ca02c", ls="--", label="SSTF (eval)")
    ax[0].axhline(res["FCFS"][0], color="#d62728", ls="--", label="FCFS (eval)")
    ax[0].set(xlabel="episode", ylabel="mean response time (ms)", yscale="log",
              title=f"Training (includes exploration) - {ep_done}/{episodes} episodes")
    ax[0].legend(fontsize=8, loc="lower left")
    ltm, k = last_train_mean()

    ax[0].text(0.98, 0.97,
               f"{'FINAL' if ep_done >= episodes else f'at ep {ep_done}'} greedy eval: {res['DQN greedy'][0]:.1f} ms\n"
               f"  (n={n_eval} workloads: spread ±{res['DQN greedy'][1]:.1f}, std. error ±{res['DQN greedy'][1] / math.sqrt(n_eval):.1f})\n"
               f"SSTF {res['SSTF'][0]:.1f} ms   FCFS {res['FCFS'][0]:.1f} ms\n"
               f"training mean, last {k} eps: {ltm:.1f} ms",
               transform=ax[0].transAxes, 
               ha="right", 
               va="top", 
               fontsize=10, 
               family="monospace",
               bbox=dict(boxstyle="round", facecolor="white", edgecolor="#555", alpha=0.92))
    
    a2 = ax[1]
    a2.plot(ep_, eps_, color="#e6550d", lw=2, label="epsilon")
    a2.set(xlabel="episode", ylabel="epsilon", title="Schedules")
    a3 = a2.twinx()
    a3.plot(ep_, lr_, color="#756bb1", lw=2, label="learning rate")
    a3.set_ylabel("learning rate")
    a2.legend(*[sum(z, []) for z in zip(a2.get_legend_handles_labels(), a3.get_legend_handles_labels())], loc="center right")

    names = list(res)
    vals = [res[n][0] for n in names]
    errs = [res[n][1] for n in names]

    ax[2].bar(names, vals, yerr=errs, capsize=4, color=["#d62728", "#2ca02c", "#08519c"])
    for i, v in enumerate(vals): 
        ax[2].text(i, v, f"{v:.0f}", ha="center", va="bottom")

    ax[2].set(ylabel="mean response time (ms)", title=f"Greedy evaluation, {n_eval} workloads (seeds {seeds.start}-{seeds.start + n_eval - 1})")
    fig.suptitle(run, fontsize=8, color="#555")
    fig.tight_layout()
    png = os.path.join(run_dir, f"ep{ep_done:04d}.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)

    return png


def save_weights(env, net, nin, nout, use_mask, tag, run, plain, args, run_dir, ep_done):
    e = env.unwrapped
    atomic(lambda f: torch.save({
        "state_dict": net.state_dict(), "episodes_done": ep_done,
        "nin": int(nin), "nout": int(nout), "hidden": 128, "use_mask": use_mask, "tag": tag, "run": run,
        "env_kwargs": plain(dict(queue_slots=e.K, episode_requests=e.N, mean_interarrival_ms=e.mean_ia,
                           read_fraction=e.read_frac, sizes=list(e.sizes), reward_mode=e.reward_mode,
                           reward_scale=e.reward_scale, invalid_penalty=e.invalid_penalty,
                           max_consecutive_invalid=e.max_invalid, wait_scale_ms=e.wait_scale)),
        "train": plain(vars(args)),
    }, f), os.path.join(run_dir, f"ep{ep_done:04d}.pt"))


def snapshot(ep_done, base_all, env, net, nin, nout, use_mask, tag, run, plain, args, 
             run_dir, seeds, snap_n, episodes, eval_hist, last, save_csv, final=False):
    """Weights + CSV first (cheap, safe), then a greedy eval and the PNG (may fail without losing the rest).greedy
    final=True evaluates on all A.eval_episodes workloads...otherwise on the first SNAP_N (cheaper)."""
    save_weights(env, net, nin, nout, use_mask, tag, run, plain, args, run_dir, ep_done)
    save_csv()

    try:
        k = len(seeds) if final else snap_n
        dq = evaluate(lambda o, m=None, r=None: greedy(o, net, use_mask, m, r), env, range(seeds.start, seeds.start + k), return_all=True)
        res = {n: ms(base_all[n][:k]) for n in ("FCFS", "SSTF")}
        res["DQN greedy"] = ms(dq)
        eval_hist.append((ep_done, res["DQN greedy"][0]))
        last.update(ep=ep_done, res=res, n=k, final=final, dq_all=dq, base_all={n: base_all[n][:k] for n in base_all})
        save_png(ep_done, res, k, episodes, run, run_dir, seeds)
    except Exception as ex:
        print(f"  [snapshot ep {ep_done}] weights+CSV saved, eval/PNG failed: {ex!r}", flush=True)


class Replay:
    """Preallocated NumPy ring buffer: O(1) add, O(batch) random sampling."""
    def __init__(self, target_net_cap, nin, nact):
        self.cap = target_net_cap
        self.n_valid_obsv = 0
        self.i = 0

        self.observations = np.zeros((target_net_cap, nin), np.float32)
        self.next_obs = np.zeros((target_net_cap, nin), np.float32)
        self.actions = np.zeros(target_net_cap, np.int64)
        self.rewards = np.zeros(target_net_cap, np.float32)
        self.done_flags = np.zeros(target_net_cap, np.float32)
        self.action_masks = np.zeros((target_net_cap, nact), bool)

    def add(self, observation, action_taken, reward, next_obs, done, mask_array):
        i = self.i
        self.observations[i] = observation
        self.actions[i] = action_taken
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.done_flags[i] = done
        self.action_masks[i] = mask_array
        self.i = (i + 1) % self.cap
        self.n_valid_obsv = min(self.n_valid_obsv + 1, self.cap)

    def sample(self, bs):
        j = np.random.randint(0, self.n_valid_obsv, bs)
        return (self.observations[j], self.actions[j], self.rewards[j], 
                self.next_obs[j], self.done_flags[j], self.action_masks[j])

    def __len__(self): 
        return self.n_valid_obsv