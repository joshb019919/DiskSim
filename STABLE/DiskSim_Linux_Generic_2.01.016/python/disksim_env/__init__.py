from gymnasium.envs.registration import register
from .env import DiskSchedEnv

register(id="DiskSched-v0", entry_point="disksim_env.env:DiskSchedEnv")
__all__ = ["DiskSchedEnv"]
register(id="DiskSchedRecycled-v0", entry_point="disksim_env.recycling:RecyclingEnv")   # same env, worker process replaced every N resets
