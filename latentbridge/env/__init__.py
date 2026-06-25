from .gridworld import GridWorld
from .lifeworld import LifeWorld
from .physicsworld import PhysicsWorld
from .desktopworld import DesktopWorld
from .simcalc import SimCalcWorld

__all__ = ["GridWorld", "LifeWorld", "PhysicsWorld", "DesktopWorld",
           "SimCalcWorld", "make_env"]


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
    if kind == "physicsworld":
        return PhysicsWorld(dims=e.get("dims", 2), dt=e.get("dt", 0.1),
                            thrust=e.get("thrust", 2.0), drag=e.get("drag", 0.10),
                            gravity=e.get("gravity", 0.0), max_steps=e.get("max_steps", 60),
                            reach_radius=e.get("reach_radius", 0.12),
                            soft_landing=e.get("soft_landing", False),
                            land_speed=e.get("land_speed", 0.4),
                            obs_mode=e.get("obs_mode", "state"),
                            img_size=e.get("img_size", 16), seed=seed)
    if kind == "desktopworld":
        return DesktopWorld(max_steps=e.get("max_steps", 12),
                            step_cost=e.get("step_cost", 0.02), seed=seed)
    if kind == "simcalc":
        return SimCalcWorld(max_value=e.get("max_value", 9),
                            max_steps=e.get("max_steps", 6), seed=seed)
    if kind == "minigrid":
        # public benchmark (gymnasium MiniGrid / BabyAI) — lazy import so the env
        # package stays importable without the optional `minigrid` dependency.
        from .minigrid_adapter import MiniGridAdapter
        return MiniGridAdapter(env_id=e.get("env_id", "MiniGrid-Empty-5x5-v0"),
                               seed=seed, max_steps=e.get("max_steps"),
                               fully_obs=e.get("fully_obs", True))
    raise ValueError(f"unknown env kind: {kind}")
