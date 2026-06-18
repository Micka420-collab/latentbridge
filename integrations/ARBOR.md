# Brancher Arbor (RUC-NLPIR/Arbor) sur LatentBridge

Arbor = agent de recherche autonome (Coordinator + Executor, hypothesis tree,
chaque expérience dans un git worktree isolé, éval sur held-out). Notre
`loop/random_search.py` fait déjà la même boucle en plus simple ; Arbor est
l'**upgrade** : il *raisonne* sur les résultats au lieu d'échantillonner au hasard.

## Le contrat (déjà respecté par ce repo)

Arbor a juste besoin de : **une commande qui tourne + un nombre à maximiser**.

- Commande : `python -m latentbridge.experiment --config configs/base.yaml --set <clé>=<val> ...`
- Métrique : `runs/<tag>/metrics.json` → champ `score` (et `guidance_gain`, etc.)
- Espace de recherche : `loop/search_space.yaml`

## Installation

```bash
pip install arbor-agent
arbor setup          # choisir le provider LLM -> OpenRouter (clé dans .env)
```

`arbor setup` te demandera le provider. Avec OpenRouter, renseigne
`OPENROUTER_API_KEY` (déjà dans `.env`) et un modèle Coordinator. Pour Arbor qui
*raisonne* sur les expériences, prends un modèle costaud (Claude/GPT/DeepSeek) ;
le modèle gratuit `google/gemma-4-31b-it:free` suffit pour le planner du gridworld
mais sera faible comme Coordinator.

## Donner l'objectif à Arbor

Arbor lit un objectif en langage + un repo de travail. Objectif suggéré :

> « Maximise `score` dans `runs/<tag>/metrics.json` en modifiant SEULEMENT les
> clés listées dans `loop/search_space.yaml` et le code de
> `latentbridge/models/`. Lance chaque essai avec
> `python -m latentbridge.experiment --config configs/base.yaml --set ...`.
> Garde toujours un essai de contrôle `loss.align=0` comme groupe témoin.
> Ne touche pas à `latentbridge/evaluate.py` (sinon tu triches sur la métrique). »

Le dernier point est crucial : **interdire à l'agent de modifier la fonction
d'évaluation**, sinon il « optimisera » en cassant la mesure. Arbor isole chaque
essai dans un worktree git, donc protège déjà `main`.

## Au-delà des hyperparams : laisser Arbor inventer de l'architecture

Pour une vraie découverte d'architecture (pas juste du tuning), autorise Arbor à
éditer `latentbridge/models/bridge.py` et `world_model.py` (nouveau type
d'adapter, attention, codebook discret type VQ, dynamique récurrente...). Garde
`evaluate.py` et `env/` figés comme juge neutre.
