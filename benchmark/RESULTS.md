# LatentBridge — résultats du benchmark

Held-out, CPU. `guidance_gain` = succès avec le pont − sans le pont (le cœur de la preuve).

| env | score | success_guided | success_unguided | ablation_success | guidance_gain | align_cosine |
|---|---|---|---|---|---|---|
| gridworld | 0.567 | 0.74 | 0.12 | 0.1 | 0.62 | 1.0 |
| lifeworld | 0.883 | 0.983 | 0.6 | 0.433 | 0.383 | 0.9996 |
| physics2d | 0.886 | 0.98 | 1.0 | 1.0 | -0.02 | 0.8702 |
| physics3d | 0.572 | 0.84 | 0.72 | 0.96 | 0.12 | 0.8398 |
| desktopworld | 0.96 | 1.0 | 0.233 | 0.2 | 0.767 | 0.9997 |
| simcalc | 0.99 | 1.0 | 0.117 | 0.1 | 0.883 | 0.9999 |

Reproduire : `python benchmark/run_benchmark.py --ablation`
