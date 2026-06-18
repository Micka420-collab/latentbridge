# Brancher Arbor (RUC-NLPIR/Arbor) — recherche d'archi en continu

Arbor = agent de recherche autonome (Coordinator + Executor, *hypothesis tree*,
chaque expérience dans un **git worktree** isolé, éval sur held-out). C'est
l'upgrade « qui raisonne » de notre `loop/random_search.py`.

## Déjà préparé pour toi sur ce poste

- ✅ Repo Arbor cloné : `external/arbor/`
- ✅ Le projet est un **repo git** avec un commit baseline (Arbor en a besoin)
- ✅ **`research_config.yaml`** écrit à la racine = le Contrat de Recherche (objectif,
  commande d'expérience, métrique `score`, fichiers interdits = le juge neutre)
- ✅ Provider : ta clé **OpenRouter** (OpenAI-compatible via LiteLLM) dans `.env`

## Ce qu'il te reste (install + lancement — à faire toi-même)

> L'install auto a été bloquée par le garde-fou anti-supply-chain (normal pour
> un paquet tiers). Lance-la toi-même — tape ces lignes (le préfixe `!` les exécute
> dans la session) :

```powershell
# 1) installer Arbor (paquet officiel PyPI du repo RUC-NLPIR/Arbor)
uv tool install arbor-agent --python 3.11
#    (ou, depuis les sources déjà clonées :)
#    uv pip install -e external/arbor

# 2) configurer le provider (choisis OpenAI-compatible -> base_url OpenRouter, colle la clé)
arbor setup

# 3) vérifier
arbor doctor

# 4) lancer la recherche d'architecture sur CE projet
arbor --config research_config.yaml
#    petit essai : ajoute --max-cycles 3
```

`arbor setup` (OpenRouter) : provider **OpenAI-compatible**, base_url
`https://openrouter.ai/api/v1`, clé = `OPENROUTER_API_KEY`, modèle Coordinator
costaud (Claude/GPT/DeepSeek) — le gratuit `google/gemma-4-31b-it:free` est trop
faible comme Coordinator.

Pendant un run : `/status`, `/tree`, `/evidence`, `/cost`, `/pause`, `/report`, `/abort`.

## Garde-fous scientifiques (déjà dans research_config.yaml)

- Arbor PEUT régler les boutons de `loop/search_space.yaml` et éditer
  `latentbridge/models/` (inventer de l'archi).
- Il NE DOIT PAS toucher `latentbridge/evaluate.py` ni `latentbridge/env/`
  (le juge neutre — sinon il triche sur la métrique). Les worktrees git protègent
  déjà `main`.
- À chaque cycle, un **contrôle** `align=0` est exigé comme témoin.

## Alternative immédiate (sans Arbor)

`python loop/random_search.py --trials 0` fait déjà tourner la recherche en continu.
Arbor = la version qui raisonne ; le loop = la version qui marche tout de suite.
