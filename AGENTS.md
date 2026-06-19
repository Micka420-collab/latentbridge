# AGENTS.md — comment améliorer LatentBridge SANS rien casser

Ce dépôt = le projet perso LLM × world-model (**LatentBridge**) + le planificateur
de journée. Si tu es un agent IA chargé d'**améliorer** ce projet, ces règles sont
**non négociables**.

## Directive n°1 : mesurer, jamais deviner
« Meilleur » = un `score` plus haut dans `runs/<tag>/metrics.json` sur held-out.
N'annonce jamais une amélioration que tu n'as pas mesurée.

Lancer une expérience (utilise le python du venv en chemin absolu) :
```
F:/newtechno/.venv/Scripts/python.exe -m latentbridge.experiment --config configs/base.yaml --tag <unique>
F:/newtechno/.venv/Scripts/python.exe -m latentbridge.experiment --config configs/life.yaml  --tag <unique>
```
Mesure rapide avec verdict vs baseline :
```
F:/newtechno/.venv/Scripts/python.exe scripts/measure.py --config configs/life.yaml --set loss.align=0.5
```

## Règles de sécurité (dures)
1. **JAMAIS sur `main`.** Toujours : `git checkout -b improve/<sujet>`.
2. **NE TOUCHE PAS au juge neutre** : `latentbridge/evaluate.py`,
   `latentbridge/env/`, ni les réglages d'éval dans `configs/`. Modifier le juge
   = tricher, pas améliorer.
3. Tu PEUX éditer : `latentbridge/models/` (l'architecture), les poids de pertes
   et boutons dans `configs/` et `loop/search_space.yaml`.
4. **Garde un changement seulement s'il bat la baseline** sur `score`. Sinon
   `git restore` / abandonne la branche.
5. Toujours un **contrôle** `loss.align=0 loss.ground=0 control.guide_weight=0`.
6. Ne touche jamais : `.env` (secrets), `runs/`, `.venv/`, `external/`, `mydata/today.yaml`.
7. Quand tu as une vraie amélioration : **commit sur la branche puis STOP**. Laisse
   l'humain relire le diff et merger. Ne merge pas sur `main` toi-même.

## Repères (scores à battre)
- lifeworld : score ~**0.88** (configs/life.yaml)
- gridworld : meilleur ~**0.57** (voir `loop/results.jsonl`)

## Ton rapport final
Dis : ce que tu as changé, `score baseline → nouveau score`, et le **nom de la
branche** git. Court et factuel.
