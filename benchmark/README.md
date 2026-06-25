# LatentBridge Benchmark

A small, **reproducible, honest** benchmark for the LatentBridge idea: couple a
**frozen LLM** to a **trained world model** through a learned **latent bridge**,
and *plan* (model-predictive control) inside the world model toward a goal the
LLM expresses **in language**.

## What it measures (and why it's the right metric)

In every environment the **goal is hidden from the observation** and given **only
in language**. So the *only* way to solve the task is the
`LLM → bridge → world-model` channel. That makes the bridge's value **directly
measurable**:

- **`guidance_gain` = success_with_bridge − success_without_bridge** (held-out seeds).
- A **no-bridge ablation** (`align = ground = cycle = 0`, `guide_weight = 0`)
  is run as the scientific control — if the bridge weren't doing the work, the
  ablation would match it.

Higher `guidance_gain` and a large gap to the ablation = the LLM↔world-model
coupling genuinely helps. `align_cosine` reports how well the latent space is
tied to language on held-out states.

## Environments

| env | world the model must learn |
|---|---|
| **gridworld** | structure + language goals (the proving ground) |
| **lifeworld** | a synthetic *day* (agenda/energy/habits) |
| **physics2d / physics3d** | continuous dynamics: momentum, inertia, gravity, soft-landing |
| **desktopworld** | a **GUI desktop** (apps/files/settings) — model-based, not reactive |
| **simcalc** | a calculator — used to drive the **real** Windows Calculator from language |

> Honest frontier (not in the auto-table): **vision** (`configs/physics_vision.yaml`)
> — acting from raw pixels plateaus at ~30% control. The lever is a recurrent
> latent dynamics (RSSM); see the project's RSSM results (drift halved, long-horizon
> control 0.90 vs 0.78 for the feedforward model).

## Reproduce

```bash
python -m pip install -r requirements.txt          # torch (CPU is enough) + numpy + pyyaml
python benchmark/run_benchmark.py --ablation        # full table with the no-bridge control
python benchmark/run_benchmark.py --only desktopworld,simcalc
```

Results are written to [`RESULTS.md`](RESULTS.md) and `results.json`. All runs are
held-out and CPU (small models, tens of seconds each). The methodology is
**measure-don't-guess**: no improvement is claimed that isn't measured, and a
control group runs every time.

## Reading the results (honest, incl. negatives)

The bridge is **decisive where the goal is hidden and symbolic** — language is the
only channel to know it — and the no-bridge **ablation collapses**:
`gridworld +0.62`, `lifeworld +0.38`, `desktopworld +0.77`, `simcalc +0.88`
(ablation success 0.10–0.43). That is the central claim, measured.

In **continuous physics with a dense reward**, the goal is largely inferable from
the reward/observation, so the bridge is **neutral** (`physics2d −0.02`) to
**counter-productive** (`physics3d`: ablation 0.96 **>** guided 0.84). Honest
conclusion: **language-steered, model-based planning matters when the goal cannot
be read off the observation/reward — not for dense-reward control.** Knowing
*when a method does not help* is part of the result.

## Honesty notes

- These are **small-scale** results on modest hardware; they demonstrate the
  *mechanism* (language-steered model-based control), not state-of-the-art scale.
- The bridge mechanism itself is a small learned projection — the contribution is
  the **assembly** (frozen LLM + language→latent bridge + MPC in a learned world
  model) and its measurement, not a single new layer.
