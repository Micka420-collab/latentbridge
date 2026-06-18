# LatentBridge — LLM × World Model, sur petite config

Une architecture hybride **LLM (gelé) + world model (entraîné)** qui partagent une
**interface latente apprise** (le « pont »). Conçue pour tourner sur **petite
config** (ici : RTX 3070 Ti 8 Go, CPU OK) et servir de socle à un **assistant de
vie local**.

> Objectif honnête du projet : ce n'est **pas** « créer une AGI en lançant une
> boucle ». C'est un **banc de recherche d'architecture automatisé** : on définit
> un espace d'architectures + une tâche mesurable + une métrique, et une boucle
> explore en continu pour trouver la meilleure variante. Ce qui rend le projet
> réel, c'est la métrique — pas le slogan.

---

## L'idée : le « Latent Bridge »

```
 observation ─► Encoder ─► z_t ─► WorldModel ─► z_{t+1}, reward   (dynamique apprise)
                  │  ▲                                            self-supervised
              Decoder (reconstruction, ancre z dans l'obs)
                  │
            ┌─────┴───────── LatentBridge (la pièce nouvelle) ─────────┐
            │  to_lang(z)   : latent → espace langage  (lisible par LLM) │
            │  from_lang(e) : langage → latent  (but du LLM → world model)│
            └───────────────────────────┬──────────────────────────────┘
                                         │
                       LLM gelé (planner) : donne le but en langage
```

5 pertes entraînées ensemble (poids = l'espace de recherche) :

| Perte | Rôle |
|---|---|
| `dyn` | prédire le prochain latent (world model) |
| `rew` | prédire la récompense |
| `recon` | reconstruire l'observation (ancrage du latent) |
| `align` | aligner `to_lang(z)` sur l'embedding langage de l'état |
| `ground` | `from_lang(texte)` → latent (le LLM passe ses buts en langage) |
| `cycle` | garder le pont inversible |

**Pourquoi la tâche prouve quelque chose** : dans le gridworld, le **but n'est pas
visible dans l'observation** — il est donné *uniquement en langage*. L'agent ne
peut donc l'atteindre qu'en passant par `LLM → from_lang → world model`. On mesure
directement `guidance_gain = succès(avec pont) − succès(sans pont)`. Un groupe de
contrôle `align=0` tourne automatiquement à chaque génération : « le pont aide »
est **mesuré, pas supposé**.

Résultat du baseline (CPU, ~30 s) : succès **0.18 → 0.48** avec le pont, alignement
latent↔langage **0.9999**.

---

## Démarrage rapide

```bash
# 1. venv déjà créé dans .venv (Python 3.14 + torch CPU)
.venv\Scripts\activate

# 2. une expérience
python -m latentbridge.experiment --config configs/base.yaml --tag test1

# 3. la boucle de recherche continue (le cœur du projet)
python loop/random_search.py --trials 30      # ou --trials 0 = à l'infini
```

Chaque run écrit `runs/<tag>/metrics.json` (contrat avec la boucle) :
`score`, `success_rate_guided/unguided`, `guidance_gain`, `align_cosine_heldout`...

L'espace de recherche est dans `loop/search_space.yaml`. Le baseline / défauts
d'architecture sont dans `configs/base.yaml`.

---

## GPU (pour scaler)

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Puis `--device cuda`. Le gridworld n'en a pas besoin ; ça sert quand on passe à
des mondes plus gros (pixels) ou à un world model plus large.

---

## Feuille de route vers l'assistant de vie

1. **(fait)** Pont validé sur gridworld : but en langage → world model.
2. **World model personnel** : remplacer le gridworld par *ton monde* — séquences
   d'événements (agenda, tâches, habitudes, fichiers). Le world model apprend à
   prédire ta journée ; le LLM planifie/assiste via le pont. Même architecture,
   nouveau `env/`.
3. **LLM 100 % local** : basculer le planner de OpenRouter vers **Ollama** (un
   modèle ~7B quantifié tient dans 8 Go) pour un assistant privé et offline.
   L'abstraction `latentbridge/llm/` est déjà prête pour ce swap.
4. **Boucle continue Arbor + Hermes** : voir `integrations/`.

---

## Sur « battre l'existant »

Voir `integrations/BENCHMARKING.md`. En résumé : pour affirmer « plus puissant que
ce qui existe », il faut (a) un benchmark public, (b) des baselines de référence
implémentées, (c) une comparaison à budget de calcul égal. La boucle est l'outil ;
la rigueur de comparaison est ce qui rend la conclusion crédible.
