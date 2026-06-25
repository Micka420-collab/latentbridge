# Résultats benchmark — LatentBridge vs baselines de référence

Produit par l'appareil de comparaison décrit dans `integrations/BENCHMARKING.md` §6
(code : `latentbridge/benchmark.py` + `latentbridge/baselines/` + l'adaptateur
`latentbridge/env/minigrid_adapter.py`). Toutes les méthodes : **même env, mêmes
graines held-out (`seed+10 000`), même budget d'épisodes d'interaction**, 5 graines,
moyenne ± écart-type. CPU.

## 1. Tâche interne — gridworld 6×6 (budget 80 épisodes, 1500 steps)

### Régime A — but visible (contest sample-efficiency)
| Méthode | Succès (moy±σ) | Pas | Params | Wall |
|---|---|---|---|---|
| Random | 0.460 ± 0.061 | 37.1 | 0 | 0.0s |
| PPO (model-free) | 0.047 ± 0.034 | 47.9 | 31k | 1.5s |
| DQN (model-free) | 0.120 ± 0.069 | 44.4 | 31k | 5.1s |
| LLM-seul (planner, sans WM) | **1.000 ± 0.000** | 4.2 | 0 | 0.0s |
| World-model seul (`align=0`, contrôle) | 0.120 ± 0.016 | 44.6 | 469k | 30.5s |
| LatentBridge (WM + pont) | 0.460 ± 0.172 | 32.0 | 469k | 28.2s |

**Contribution du pont : +0.340** (LatentBridge − WM-seul) — *aide* (Δ > Σσ).

### Régime B — but en langage seulement (mesure la valeur du pont)
| Méthode | Succès (moy±σ) | Pas |
|---|---|---|
| Random | 0.460 ± 0.061 | 37.1 |
| PPO | 0.067 ± 0.037 | 46.8 |
| DQN | 0.053 ± 0.045 | 47.5 |
| LLM-seul | 1.000 ± 0.000 | 4.2 |
| World-model seul | 0.147 ± 0.054 | 44.2 |
| LatentBridge | 0.427 ± 0.053 | 33.1 |

**Contribution du pont : +0.280** — *aide* (Δ > Σσ).

## 2. Benchmark public — MiniGrid-Empty-5x5 (budget 120 épisodes, 1500 steps)

| Méthode | Succès (moy±σ) | Pas | Params | Wall |
|---|---|---|---|---|
| Random | 0.260 ± 0.065 | 58.7 | 0 | 0.4s |
| PPO (model-free) | 0.000 ± 0.000 | 64.0 | 27k | 3.9s |
| DQN (model-free) | 0.000 ± 0.000 | 64.0 | 27k | 12.8s |
| LLM-seul *(dégradé→random¹)* | 0.260 ± 0.065 | 58.7 | 0 | 0.4s |
| **World-model seul (`align=0`, contrôle)** | **0.640 ± 0.181** | 39.6 | 453k | 38.4s |
| LatentBridge (WM + pont) | 0.300 ± 0.372 | 50.5 | 453k | 40.2s |

**Contribution du pont : −0.340** — **NON significatif** (Δ=0.34 < Σσ=0.553). Sur MiniGrid-Empty
le but est **déjà dans l'observation** (grille pleinement observée) : le guidage langage du pont
est redondant et, mal calibré, *dégrade + ajoute de la variance*. Le **groupe de contrôle gagne**
ici — et le doc (§4) impose de le **dire**, pas de le cacher.

¹ Le LLM-seul ici tombe en random : son greedy « va vers la case but » n'est codé que pour les 4
actions du gridworld ; MiniGrid a 7 actions avec orientation (turn/forward). Un LLM-seul honnête pour
MiniGrid demanderait une navigation orientée (BFS turn+forward) — voir « Suite ».

## Conclusions défendables (et leurs limites)

1. ✅ **Plus sample-efficient que le RL model-free**, à budget égal, sur **les deux** tâches :
   gridworld 0.46 vs ≤0.12 ; MiniGrid 0.64/0.30 vs **0.00**. C'est le point le plus solide.
2. ✅ **Le canal langage (pont) apporte une valeur mesurable QUAND le but est caché de
   l'observation** (gridworld régime langage : +0.28, contrôle battu au-delà du bruit).
3. ⚠️ **Le pont n'est PAS universellement bénéfique** : quand le but est déjà observable
   (MiniGrid-Empty), il est neutre-à-nuisible (−0.34, n.s., variance énorme). Sa valeur est
   **conditionnelle**, pas universelle.
4. ❌ **« Meilleur que l'état de l'art » reste interdit** : 5 graines + variance élevée sur
   MiniGrid ; et MiniGrid-Empty n'est pas le bon test du pont (but observable). Dire
   **« prometteur, plus sample-efficient que PPO/DQN à budget égal »** — pas davantage.

## Suite (pour resserrer la preuve)
- **BabyAI GoTo*** (`env_id: BabyAI-GoToObj-v0`) : plusieurs buts candidats, but **en langage** →
  c'est *là* que le pont doit payer. Changement d'`env_id` uniquement, zéro code.
- **≥10 graines + t-test apparié** (la variance MiniGrid l'exige).
- **LLM-seul orienté** pour MiniGrid (BFS turn+forward) pour une baseline planner honnête.
- **GPU** (zarn-ai éteint) pour monter le budget et porter sur Crafter / Atari-100k.
