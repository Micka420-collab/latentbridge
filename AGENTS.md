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

## Niveaux d'autorisation
- **Niveau 1** (sûr) : boutons dans `configs/` et `loop/search_space.yaml`.
- **Niveau 2** : l'architecture dans `latentbridge/models/`.
- **Niveau 3** (risqué) : le **noyau du loop** — `latentbridge/train.py`,
  `latentbridge/controller.py`, `loop/`. Autorisé, MAIS snapshot + branche +
  **validation à deux** obligatoires. Jamais merger seul.

**Hors-limites à TOUS les niveaux — le juge neutre** : `latentbridge/evaluate.py`,
`latentbridge/env/`, et les réglages d'éval dans `configs/`. Raison : si tu édites
le juge, le `score` ne veut plus rien dire — ce n'est pas un risque de casse,
c'est que tes mesures deviennent fausses. Jamais non plus : `.env`, `runs/`,
`.venv/`, `external/`, `mydata/today.yaml`.

## Sauvegarde / restauration (OBLIGATOIRE à chaque modif)
Avant CHAQUE modif, prends un point de restauration :
```
F:/newtechno/.venv/Scripts/python.exe scripts/snapshot.py save "<ce que tu tentes>"
```
Si la mesure dit pire ou inutile, reviens en arrière :
```
F:/newtechno/.venv/Scripts/python.exe scripts/snapshot.py restore <id>
F:/newtechno/.venv/Scripts/python.exe scripts/snapshot.py list      # voir les points
```
(Hermes snapshote aussi TOUT automatiquement avant chaque écriture — `/rollback`
dans le chat pour annuler.)

## La boucle d'amélioration (suis-la)
1. `git checkout -b improve/<sujet>` — jamais sur `main`.
2. `snapshot.py save "..."` → une modif ciblée → `measure.py` → un contrôle `align=0`.
3. **Mieux** que la baseline ? commit sur la branche. **Pire/inutile** ? `snapshot.py restore`.
4. Quand tu tiens une vraie amélioration : **STOP — on valide ENSEMBLE** avant de
   merger sur `main`. Jamais seul.

## Repères (scores à battre)
- lifeworld : score ~**0.88** (configs/life.yaml)
- gridworld : meilleur ~**0.57** (voir `loop/results.jsonl`)

## Ton rapport final
Dis : ce que tu as changé, `score baseline → nouveau score`, et le **nom de la
branche** git. Court et factuel.
