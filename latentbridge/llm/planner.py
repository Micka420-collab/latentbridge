"""The frozen-LLM planner.

Its job is to turn a free-form intention ("I'm exhausted, just want to rest")
into a concrete goal the world model can plan toward. The env owns the canonical,
in-distribution target description; the LLM only *picks which goal* (a language ->
goal classification). The bridge then maps that goal text into a latent target.

Two backends:
  - "mock":       offline. Reads the env's true goal directly. Lets the whole
                  pipeline run with no network in the hot loop.
  - "openrouter": real frozen LLM via the OpenAI-compatible OpenRouter API. Used
                  when the env exposes a goal menu (e.g. LifeWorld).

propose() always returns {"subgoal_text": <canonical target>, "goal_key": <key>}.
It never raises in the control loop: any backend error falls back to mock.
"""
from __future__ import annotations

import os


class Planner:
    def __init__(self, backend: str = "mock", model: str | None = None, size: int = 6):
        self.backend = backend
        self.size = size  # kept for backward compat / gridworld
        self.model = model or os.environ.get("PLANNER_MODEL", "google/gemma-4-31b-it:free")
        self._client = None
        if backend == "openrouter":
            self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            self._client = OpenAI(
                api_key=key,
                base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            )
        except Exception as e:
            print(f"[planner] openrouter init failed ({e}); falling back to mock")
            self.backend = "mock"

    # ------------------------------------------------------------------ #
    def propose(self, env, intention: str | None = None) -> dict:
        # Use the LLM only when the env offers a discrete goal menu to classify.
        if (self.backend == "openrouter" and self._client is not None
                and hasattr(env, "goal_menu")):
            key = self._classify_llm(env, intention)
            if key is not None:
                return {"subgoal_text": env.goal_state_text(key), "goal_key": key}
        # mock / fallback: the env reveals its true goal
        key = env.true_goal_key()
        return {"subgoal_text": env.goal_state_text(key), "goal_key": key}

    def _classify_llm(self, env, intention=None):
        menu = env.goal_menu()
        text = intention or env.goal_instruction()
        options = "\n".join(f"- {k}: {desc}" for k, desc in menu)
        prompt = (
            f"User says: \"{text}\"\n\n"
            f"Pick the single goal that best matches, from:\n{options}\n\n"
            f"Reply with ONLY the goal key (one word)."
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8, temperature=0.0,
            )
            out = (resp.choices[0].message.content or "").strip().lower()
            for k, _ in menu:
                if k in out:
                    return k
        except Exception as e:
            print(f"[planner] llm classify failed ({e}); using mock goal")
        return None
