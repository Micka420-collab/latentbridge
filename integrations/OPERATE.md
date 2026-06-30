# How Shellia operates Windows apps (the playbook)

Shellia drives real apps through the **UIA accessibility API** — acting on controls **by
name / AutomationId** (reliable `Invoke` + `SetValue`), not blind coordinate clicks. Tool:
`scripts/operate_app.py` (wraps `latentbridge/env/uia.py`).

## The loop: PERCEIVE → DECIDE → ACT → VERIFY

```
windows                      # what apps are open?
inspect "<title>"            # what named controls can I act on in this one?
click  "<title>" --name X    # act on the control called X (or --id <AutomationId>)
settext "<title>" --role Document --text "…"
read   "<title>"             # re-perceive to confirm the effect (number / values)
```

Always **re-perceive after acting** — that's how Shellia knows the action worked instead of
hoping it did.

## Worked example (real run, French Windows — locale-proof via AutomationId)

```
$ operate_app.py demo
1) PERCEIVE: opened 'Calculatrice' — menu: Button="Fermer Calculatrice", Text="L’affichage est 0", …
2) ACT (UIA Invoke by AutomationId): 7(num7Button) ok | +(plusButton) ok | 7 ok | =(equalButton) ok
3) VERIFY (re-perceive): « L’affichage est 14 »   ✅
```

Shellia perceived the app, acted by control name, and read back the result — no pixels, no
guessing coordinates.

## Recipes (app-agnostic verbs, same everywhere)

| goal | commands |
|---|---|
| survey the desktop | `windows` |
| see what's clickable | `inspect "Discord"` |
| open an app / URL | `launch "notepad"` · `launch "https://polymarket.com"` |
| press a named button | `click "Calc" --id equalButton` · `click "App" --name "Save"` |
| fill a text field | `settext "Notepad" --role Document --text "hello"` |
| read current state | `read "Calc"` (prominent number + named values) |

**Why by-name beats coordinates**: control names/IDs survive window moves, resolution
changes, and theme changes; `Invoke` fires the control's real action even when it's not
visible at a pixel. `extract_number` + value-reading let Shellia *verify* outcomes.

## Architecture (honest)

The "hands" run **Windows-side** (UIA is local to the desktop at 192.168.1.48). Shellia's
brain is on the **VM** (192.168.1.112). Today the brain can *see* the desktop (perception
pushed to `live_anticipation.json`) and, once Micka arms winagent's control client, *click/
type/launch* via coordinates. To give the brain these **reliable by-name UIA verbs remotely**,
the next step is a Windows-side executor (a winagent `/operate` endpoint that runs these
verbs) — **arming that remote surface is Micka's gesture**, exactly like the control client.

Until then, `operate_app.py` is run **on the Windows machine** (where this demo ran), and the
playbook above is the method Shellia follows.
