# Desktop anticipation → Shellia (VM)

How Shellia (the hermes agent on the VM) knows **what Micka is doing on his PC and what
he's about to do** — without any screenshot or raw text leaving the Windows machine.

## Topology (perception is Windows-side, brain is VM-side)

```
Windows (192.168.1.48)                         VM (192.168.1.112)
─────────────────────                          ──────────────────
UIA accessibility tree                         hermes agent (the brain)
  → featurize_predictable  (privacy:             reads on demand:
     control-TYPE histogram, no text)              scripts/shellia_anticipation.py
  → app_world_model (enc/WM/dec)                   └─ reads live_anticipation.json
  → imagine next state
  → bridge.to_lang → LANGUAGE
  → scripts/anticipate_report.py  ── scp ──▶   /root/latentbridge/live_anticipation.json
```

Only a tiny JSON crosses the network: `{ts, window, now, anticipated, now_controls, grow}`.
No pixels, no keystrokes, no document text — just control-type structure mapped to language.

## Data plane (additive, breaks nothing)

- **Windows** runs `scripts/anticipate_report.py` (schedule via schtasks for continuous
  anticipation). It perceives, writes `runs/app_world_model/live_anticipation.json`, and
  `scp`s it to the VM.
- **VM** receives it at `/root/latentbridge/live_anticipation.json`.

## How Shellia consumes it

The hermes agent has a terminal tool. To know what Micka is up to, it runs:

```bash
python3 /root/latentbridge/scripts/shellia_anticipation.py
```

→ `Micka — fenetre active: … / maintenant: … / va probablement: …`. The reader reports
staleness honestly (frais / périmé) so Shellia never trusts a dead file.

**No running service is modified.** hermes-brain / hermes-gateway / latentbridge are
untouched. To make consultation *automatic* (vs on-demand), add one line to Hermes memory
telling it to consult the reader when reasoning about Micka's activity — a Micka gesture,
not an agent self-edit.

## Provenance

- Bridge trained on the real `app_world_model` by `scripts/train_app_bridge.py`
  (240 real desktop (obs,text) pairs, 13 states, align cosine 0.987).
- Anticipation chain validated in EXP-007 (×2 next-action in sim; live language on desktop).
