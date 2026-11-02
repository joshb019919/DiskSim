"""Minimal DQN using only the Gymnasium API (no DiskSim-specific code).

Uses info["action_mask"] (valid queue slots) for exploration, the greedy choice and the Bellman target
(--no-mask turns that off). Every hyperparameter is a command-line option: see  --help.

Output (under --out-dir, default results/):
  <RUN>/epNNNN.png, epNNNN.pt   PNG + weights every --snap-every episodes (RUN = hyperparameter tag + timestamp)
  <RUN>/summary.json            final metrics + hyperparameters (read by main.py)
  <RUN>_history.csv             per-episode history of the whole run
"""
import sys, os, random, time, math, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, gymnasium as gym
from baselines import fcfs, sstf, evaluate


from utils import (atomic, 
                   last_train_mean,
                   ms,
                   optimizer, 
                   snapshot,
                   EVAL_HIST, 
                   HERE, 
                   HIST, 
                   LAST, 
                   NEG, 
                   Replay)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("episodes", nargs="?", type=int, default=50, help="training episodes (positional, or use --episodes)")
    ap.add_argument("--episodes", dest="episodes_opt", type=int, default=None, help="same as the positional episodes")
    g = ap.add_argument_group("exploration (epsilon: 1.0 for HOLD, linear decay over DECAY, then EPS_END)")
    g.add_argument("--hold", type=float, default=0.25, help="fraction of episodes at epsilon=1.0")
    g.add_argument("--decay", type=float, default=0.50, help="fraction of episodes spent decaying epsilon")
    g.add_argument("--eps-end", type=float, default=0.02, help="final epsilon")
    g = ap.add_argument_group("optimisation")
    g.add_argument("--lr0", type=float, default=5e-4, help="learning rate at the start (cosine decay)")
    g.add_argument("--lr1", type=float, default=5e-5, help="learning rate at the end")
    g.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay for the optimizer")
    g.add_argument("--optim", type=optimizer, default=torch.optim.AdamW, help="optimizer for the network")
    g.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer, ignored otherwise")
    g.add_argument("--policy-dropout", type=float, default=0.0, help="Dropout rate for the policy network")
    g.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    g.add_argument("--batch", type=int, default=64, help="minibatch size")
    g.add_argument("--train-every", type=int, default=4, help="one gradient step every N environment steps")
    g.add_argument("--warmup", type=int, default=1000, help="transitions in the buffer before training starts")
    g.add_argument("--tgt-update", type=int, default=1000, help="copy online -> target network every N environment steps")
    g.add_argument("--replay-cap", type=int, default=100_000, help="replay buffer capacity (transitions)")
    g.add_argument("--steps", type=int, default=1000, help="initial value of the global step counter (shifts the update cadence)")
    g = ap.add_argument_group("run / environment")
    g.add_argument("--seed", type=int, default=0, help="agent seed (init, exploration, sampling) and training-workload seeds")
    g.add_argument("--no-mask", action="store_true", help="do not use the action mask")
    g.add_argument("--reward-mode", default=None, choices=["response", "service", "queue_time"], help="environment reward (default: the env's)")
    g.add_argument("--reward-scale", type=float, default=None, help="environment reward scale (default: the env's)")
    g.add_argument("--eval-episodes", type=int, default=10, help="workloads (seeds 1000..) for the FINAL greedy evaluation; more = less noise")
    g.add_argument("--snap-eval-episodes", type=int, default=10, help="workloads for each intermediate snapshot's evaluation (a prefix of the final set)")
    g.add_argument("--recycle-every", type=int, default=250, help="replace the DiskSim worker process every N resets (DiskSim leaks ~0.5 MB per reset); 0 = run in-process")
    g.add_argument("--snap-every", type=int, default=25, help="save PNG + weights (and run a greedy eval) every N episodes")
    g.add_argument("--out-dir", default=os.path.join(HERE, "..", "results"), help="where runs are written")
    g.add_argument("--threads", type=int, default=None, help="torch CPU threads (1 is best when running several jobs)")
    return ap.parse_args()


A = parse_args()

env_kwargs = {}
if A.reward_mode: 
    env_kwargs["reward_mode"] = A.reward_mode
if A.reward_scale: 
    env_kwargs["reward_scale"] = A.reward_scale

env = gym.make("DiskSchedRecycled-v0", recycle_every=A.recycle_every, **env_kwargs) \
    if A.recycle_every > 0 \
    else gym.make("DiskSched-v0", **env_kwargs)

EPISODES = A.episodes_opt if A.episodes_opt is not None else A.episodes
USE_MASK = not A.no_mask

# Per-run identity: TAG encodes the hyperparameters
# RUN adds a timestamp so every run gets its own folder.
TAG = (f"dqn_{EPISODES}ep_rew-{env.unwrapped.reward_mode}x{env.unwrapped.reward_scale:g}"
       f"_eps-h{A.hold:g}d{A.decay:g}_wd{A.weight_decay:g}_e{A.eps_end:g}_lr-{A.lr0:g}to{A.lr1:g}"
       f"_g{A.gamma:g}_bs{A.batch}_te{A.train_every}_tu{A.tgt_update}_rc{A.replay_cap // 1000}k"
       f"_mask{int(USE_MASK)}_s{A.seed}_optim{A.optim.__name__}_pdrop{A.policy_dropout}")

OUT = os.path.abspath(A.out_dir)
RUN = f"{TAG}_{time.strftime('%Y%m%d-%H%M%S')}"
RUN_DIR = os.path.join(OUT, RUN)                     # PNG + weights snapshots + summary.json
CSV = os.path.join(OUT, f"{RUN}_history.csv")        # full-run history, flat in the out dir

PLAIN = lambda d: json.loads(json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o)))  # no numpy types
TRAIN_SEED0 = 1_000_000 * (A.seed + 1)               # training workloads never overlap the eval seeds (1000-1009)
SEEDS = range(1000, 1000 + A.eval_episodes)          # final evaluation workloads
SNAP_N = min(A.snap_eval_episodes, A.eval_episodes)  # snapshots use the first SNAP_N of them
os.makedirs(RUN_DIR, exist_ok=True)

if A.threads: 
    torch.set_num_threads(A.threads)

# Policy network dims
nin, nout = env.observation_space.shape[0], env.action_space.n
mlp = lambda p=0.0: nn.Sequential(nn.Linear(nin, 128), nn.ReLU(), nn.Dropout(p),
                                  nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p),
                                  nn.Linear(128, nout))
torch.manual_seed(A.seed)
random.seed(A.seed)
np.random.seed(A.seed)

# MLPs for policy and target network
policy_network, target_network = mlp(A.policy_dropout), mlp(A.policy_dropout)
target_network.load_state_dict(policy_network.state_dict())
policy_network.train()
target_network.eval()  # Targets must be deterministic or eval lasts forever

# Optimizer class
if A.optim is torch.optim.SGD:
    opt = A.optim(policy_network.parameters(), lr=A.lr0, weight_decay=A.weight_decay, momentum=A.momentum)
else:
    opt = A.optim(policy_network.parameters(), lr=A.lr0, weight_decay=A.weight_decay)

buf = Replay(A.replay_cap, nin, nout)
steps = A.steps


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


def save_csv():
    atomic(lambda f: np.savetxt(f, np.array(HIST, dtype=float).reshape(-1, 6), delimiter=",",
           header="episode,epsilon,lr,mean_response_ms,completed,disksim_hang", comments=""), CSV)

BASE_ALL = {n: evaluate(p, env, SEEDS, return_all=True) for n, p in [("FCFS", fcfs), ("SSTF", sstf)]}  # computed once, per workload
BASE = {n: ms(v) for n, v in BASE_ALL.items()}                                                         # over the full final set
print(f"run folder: {RUN_DIR}\nbaselines ({A.eval_episodes} workloads): "
    + f"FCFS {BASE['FCFS'][0]:.1f} ms, SSTF {BASE['SSTF'][0]:.1f} ms", flush=True)

t0 = time.time()
status = "interrupted"

try:
  for ep in range(EPISODES):
    obs, info = env.reset(seed=TRAIN_SEED0 + ep)
    done = False
    mask = info["action_mask"]
    frac = ep / EPISODES

    eps = 1.0 if frac < A.hold else max(A.eps_end, 1.0 - (1.0 - A.eps_end) * (frac - A.hold) / A.decay)
    lr = A.lr1 + 0.5 * (A.lr0 - A.lr1) * (1 + math.cos(math.pi * frac))

    for pg in opt.param_groups: 
        pg["lr"] = lr

    while not done:
        if random.random() < eps:
            a = int(np.random.choice(np.flatnonzero(mask))) if USE_MASK else env.action_space.sample()
        else:
            a = greedy(obs, policy_network, USE_MASK, mask)

        nobs, r, term, trunc, info = env.step(a)
        nmask = info["action_mask"]
        buf.add(obs, a, r, nobs, float(term), nmask)   # time-limit => bootstrap (not terminal)
        obs = nobs
        mask = nmask
        done = term or trunc
        steps = steps + 1

        if len(buf) >= A.warmup and steps % A.train_every == 0:
            o, a_, r_, n_, d_, m_ = buf.sample(A.batch)
            q = policy_network(torch.as_tensor(o)).gather(1, torch.as_tensor(a_).long()[:, None]).squeeze(1)

            with torch.no_grad():
                qn = target_network(torch.as_tensor(n_))
                if USE_MASK:
                    qn = qn.masked_fill(~torch.as_tensor(m_), NEG)
                y = torch.as_tensor(r_, dtype=torch.float32) + A.gamma * (1 - torch.as_tensor(d_, dtype=torch.float32)) * qn.max(1).values

            loss = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if steps % A.tgt_update == 0: 
            target_network.load_state_dict(policy_network.state_dict())

    HIST.append((ep, eps, lr, info['mean_response_ms'], info['completed'], float(bool(info.get('disksim_hang')))))

    if ep % 25 == 0:
        print(f"ep {ep:4d} eps={eps:.4f} lr={lr:.1e} \t train {info['mean_response_ms']:8.1f} ms "
            + f"\t hangs {int(sum(x[5] for x in HIST))}  ({time.time()-t0:.0f}s)", flush=True)
    if (ep + 1) % A.snap_every == 0:
        # base_all, env, net, nin, nout, use_mask, tag, run, plain, args, run_dir, seeds, snap_n, episodes, eval_hist, last
        snapshot(ep + 1, BASE_ALL, env, policy_network, nin, nout, USE_MASK, TAG, RUN, PLAIN, A, RUN_DIR, 
                 SEEDS, SNAP_N, EPISODES, EVAL_HIST, LAST, save_csv, final=(ep + 1 == EPISODES))

  status = "ok"
finally:   # normal end, crash or Ctrl-C: keep whatever was trained so far
    if not LAST.get("final"):
        snapshot(len(HIST), BASE_ALL, env, policy_network, nin, nout, USE_MASK, TAG, RUN, PLAIN, A, 
                 RUN_DIR, SEEDS, SNAP_N, EPISODES, EVAL_HIST, LAST, save_csv, final=True)
    hangs = int(sum(x[5] for x in HIST))

    if LAST:
        res = LAST["res"]
        print(f"--- final eval after {LAST['ep']} episodes, {LAST['n']} workloads (mask={'on' if USE_MASK else 'off'}) ---")
        for n in ("FCFS", "SSTF", "DQN greedy"):
            print(f"{n:11s} mean response {res[n][0]:8.2f} ms (spread ±{res[n][1]:.2f}, std. error ±{res[n][1] / math.sqrt(LAST['n']):.2f})")

    ltm, k = last_train_mean()
    summary = PLAIN(dict(status=status, episodes_done=len(HIST), episodes_planned=EPISODES, seconds=time.time() - t0,
                         final_greedy_ms=LAST["res"]["DQN greedy"][0] if LAST else None,
                         final_greedy_std=LAST["res"]["DQN greedy"][1] if LAST else None,
                         final_greedy_se=(LAST["res"]["DQN greedy"][1] / math.sqrt(LAST["n"])) if LAST else None,
                         n_eval=LAST.get("n"), final_snapshot=bool(LAST.get("final")),
                         final_per_seed=dict(dqn=[float(x) for x in LAST["dq_all"]],
                                             fcfs=[float(x) for x in LAST["base_all"]["FCFS"]],
                                             sstf=[float(x) for x in LAST["base_all"]["SSTF"]]) if LAST else None,
                         fcfs_ms=BASE["FCFS"][0], sstf_ms=BASE["SSTF"][0], last_train_mean_ms=ltm,
                         disksim_hangs=hangs, tag=TAG, run=RUN, run_dir=RUN_DIR, hparams=vars(A)))
    
    with open(os.path.join(RUN_DIR, "summary.json"), "w") as f: 
        json.dump(summary, f, indent=1)
    print(f"DiskSim hangs during training: {hangs}/{len(HIST)} episodes")
    print(f"saved: {RUN_DIR}  (PNG + weights every {A.snap_every} episodes, summary.json)\n       {CSV}")
