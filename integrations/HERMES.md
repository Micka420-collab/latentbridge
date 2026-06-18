# Brancher Hermes (NousResearch/hermes-agent) sur LatentBridge

Hermes = agent LLM auto-améliorant (mémoire persistante, skills, automations
cron, multi-plateforme). Ce n'est ni un world model ni un entraîneur : c'est la
**couche de contrôle / supervision** au-dessus de la boucle de recherche.

## Rôle de Hermes ici

- Lancer / arrêter / planifier la boucle de recherche (`loop/random_search.py`
  ou Arbor) via cron.
- Surveiller `loop/results.jsonl`, te notifier (Telegram/Discord/CLI) quand un
  nouveau best apparaît, résumer les tendances.
- Mémoriser les configs gagnantes comme « skills » réutilisables.
- Être le **planner en production** : c'est le même LLM (OpenRouter) que
  `latentbridge/llm/planner.py` utilise.

## Statut réel sur ce poste (déjà installé)

Hermes **v0.16.0 est installé localement** (Python 3.11.9, venv uv) dans
`external/hermes-agent/.venv`. Déjà configuré :
- ✅ Clé **OpenRouter** dans `C:\Users\micki\AppData\Local\hermes\.env` (détectée par `hermes doctor`)
- ✅ **WhatsApp** pré-réglé : `WHATSAPP_ENABLED=true`, `WHATSAPP_MODE=self-chat`,
  `WHATSAPP_ALLOWED_USERS=33781142827`
- ✅ `config.yaml` : DM inconnus ignorés (numéro privé) + config migrée v30
- ✅ Lanceur `hermes.cmd` à la racine du projet

Lance Hermes via `.\hermes.cmd <commande>` (ou ajoute le venv au PATH).

### Ce qu'il te reste (interactif — à faire toi-même dans un PowerShell)

```powershell
cd F:\newtechno
.\hermes.cmd model      # choisis OpenRouter + un modèle (la clé est déjà là)
.\hermes.cmd whatsapp   # un QR s'affiche -> scanne-le depuis ton téléphone
.\hermes.cmd gateway    # lance l'assistant ; parle-lui en t'envoyant un message WhatsApp
```

Scan du QR : WhatsApp → **Réglages → Appareils connectés → Connecter un appareil**.
Le QR expire en ~20 s ; s'il a expiré, relance `.\hermes.cmd whatsapp`.

> ⚠️ Pont **non-officiel** (Baileys) → risque de bannissement du compte WhatsApp.
> Nous Research recommande un **numéro dédié**, pas ton numéro perso.

## Skill Hermes suggérée : "babysit-search"

Crée une skill/automation qui, toutes les heures :
1. lit `loop/results.jsonl`,
2. trie par `score`, repère le best courant et le `guidance_gain`,
3. notifie si nouveau best, sinon relance N essais,
4. archive le meilleur `config.json`.

Commandes que la skill appelle :
```bash
python loop/random_search.py --trials 20
python -c "import json;rows=[json.loads(l) for l in open('loop/results.jsonl')];\
b=max(rows,key=lambda r:r['metrics']['score']);print(b['tag'],b['metrics']['score'])"
```

## Vers l'assistant de vie 100 % local

Quand tu passeras le planner sur **Ollama** (LLM local), Hermes peut aussi
pointer sur ce même modèle local → assistant privé, offline, qui pilote son
propre world model personnel. C'est la convergence des deux repos + ce cœur.
