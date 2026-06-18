from .gridworld import GridWorld
from .lifeworld import LifeWorld

__all__ = ["GridWorld", "LifeWorld", "make_env"]


def make_env(cfg, seed: int = 0):
    """Build the env named by cfg['env']['kind'] (the swappable 'body')."""
    e = cfg["env"]
    kind = e.get("kind", "gridworld")
    if kind == "gridworld":
        return GridWorld(size=e.get("size", 6), max_steps=e.get("max_steps", 50),
                         n_walls=e.get("n_walls", 0),
                         step_penalty=e.get("step_penalty", 0.01),
                         include_goal_in_obs=e.get("include_goal_in_obs", False),
                         seed=seed)
    if kind == "lifeworld":
        return LifeWorld(n_slots=e.get("n_slots", 10), max_tasks=e.get("max_tasks", 4),
                         n_high=e.get("n_high", 2), seed=seed)
    raise ValueError(f"unknown env kind: {kind}")
