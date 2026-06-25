# « Plus puissant que ce qui existe » — comment le prouver sérieusement

Tu veux une architecture **plus efficace que l'existant sur petite config**. C'est
un objectif légitime et atteignable — mais une affirmation de supériorité n'a de
valeur que si elle est **comparée proprement**. Voici le protocole.

## 1. Choisir un axe où « petite config » a un sens

Tu ne battras pas GPT-4 en capacité brute sur 8 Go. Mais on peut viser un axe
mesurable où l'efficacité prime :

- **Sample-efficiency** : atteindre X% de succès avec le moins d'interactions/épisodes.
- **Compute-efficiency** : meilleur score à budget FLOPs/temps GPU égal.
- **Param-efficiency** : meilleur score à nombre de paramètres égal.
- **Zero-shot par instruction langage** : généraliser à des buts jamais vus,
  donnés en langage (c'est exactement ce que teste le pont ici).

C'est sur **l'efficacité**, pas la taille, qu'une petite config peut gagner.

## 2. Implémenter des baselines de référence (sinon « mieux » ne veut rien dire)

Dans le même `env/` et le même `evaluate.py`, ajoute :

- **Model-free RL** (PPO/DQN) — la baseline classique.
- **World model pur** (Dreamer-like) sans canal langage → mesure l'apport du LLM.
- **LLM seul** (planner direct, sans world model) → mesure l'apport du world model.
- **Notre LatentBridge** (les deux + le pont).

À budget égal (mêmes épisodes, même temps). Le tableau de comparaison EST le
résultat scientifique.

## 3. Tâches held-out + graines multiples

- Toujours évaluer sur des graines jamais vues à l'entraînement (déjà le cas :
  `seed + 10_000`).
- Rapporter moyenne ± écart-type sur ≥5 graines d'entraînement, pas un run unique.
  (Un seul run peut être de la chance.)

## 4. Le groupe de contrôle anti-triche

`loop/search_space.yaml` lance déjà un essai `align=0` (pont désactivé) à chaque
génération. Si le meilleur run « avec pont » ne bat pas significativement le
contrôle, **le pont n'apporte rien** — et il faut le dire, pas le cacher.

## 5. Passer à un benchmark public

Quand le gridworld est maîtrisé, porter l'architecture sur un benchmark reconnu
pour pouvoir te comparer à la littérature :

- **MiniGrid / BabyAI** (instructions en langage — idéal pour le pont).
- **Crafter** (score d'accomplissements, world models y sont étudiés).
- **Atari 100k** (le standard de sample-efficiency).

Tant que tu n'es pas sur un de ces benchmarks, dis « prometteur sur notre tâche »,
pas « meilleur que l'état de l'art ». La boucle te donne la vitesse d'itération ;
la comparaison honnête te donne la crédibilité.

---

## 6. Implémenté — comment lancer la comparaison

Le protocole ci-dessus est **codé** dans le repo (rien à reconstruire) :

| Pièce | Fichier |
|---|---|
| Baseline PPO (model-free) | `latentbridge/baselines/ppo.py` |
| Baseline DQN (model-free) | `latentbridge/baselines/dqn.py` |
| Floor aléatoire + LLM-seul (planner, sans world model) | `latentbridge/baselines/policies.py` |
| World-model-seul = contrôle `align=0` | produit par le harness (bridge ablaté) |
| Harness budget-égal multi-graines (tableau moy.±écart-type) | `latentbridge/benchmark.py` |
| Portage benchmark public MiniGrid/BabyAI | `latentbridge/env/minigrid_adapter.py` + `configs/minigrid.yaml` |

Toutes les méthodes tournent dans le **même** `make_env` / les mêmes **graines held-out**
(`seed + 10_000`), au **même budget d'épisodes d'interaction** (axe sample-efficiency).
Le harness rapporte aussi le temps-mur (proxy compute) et le nombre de paramètres
(param-efficiency) — les trois axes du §1.

```bash
# tâche interne (gridworld) — les deux régimes (but visible / but en langage seul)
python -m latentbridge.benchmark --config configs/base.yaml \
    --seeds 5 --budget 80 --steps 1500 --regime both --tag grid_v1

# benchmark public (MiniGrid-Empty) — régime « but visible » = contest sample-efficiency
python -m latentbridge.benchmark --config configs/minigrid.yaml \
    --seeds 5 --budget 120 --steps 1500 --regime visible --tag minigrid_v1
```

Sortie : `runs/<tag>/REPORT.md` (tableau) + `results.json` (brut, toutes les graines).
Le harness imprime explicitement la **contribution du pont**
(LatentBridge − World-model-seul) et signale si elle n'est **pas significative** au
nombre de graines donné — au lieu de la masquer (§4). Pour MiniGrid sur des missions
BabyAI à plusieurs buts candidats (où le canal langage paie vraiment), il suffit de
changer `env_id` (ex. `BabyAI-GoToObj-v0`) — pas de changement de code.
