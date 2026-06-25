"""DesktopWorld — a simulated GUI desktop, the next embodiment rung after physics.

Same interface as the other worlds, so the whole LatentBridge pipeline (train,
eval, bridge, MPC) runs unchanged. Here the "world model" learns how a *desktop*
evolves: focusing apps, typing, saving/deleting files, toggling settings. The
observation is a STRUCTURED accessibility-tree snapshot (which app is focused,
the editor buffer, each file's presence + content, settings, selection) — NOT
pixels: this sidesteps the 8 GB pixel-world-model budget and the fine-transition
fidelity wall, exactly as the SOTA research (June 2026) recommends.

The TASK (what the user wants) is the goal — given ONLY in language, never in the
observation — so the agent must use the frozen-LLM -> bridge -> world-model
channel to know *what* to do. Success is checked PROGRAMMATICALLY (deterministic
state assertions), not by a VLM judge — the honest, un-gameable metric.

This is the SIMULATION rung. Once the architecture proves out here, step() is
swapped to drive the REAL Windows desktop via winagent (the desktop's
"sim-to-real"); the env interface stays identical.
"""
from __future__ import annotations

import numpy as np

APPS = ["fichiers", "editeur", "reglages"]
FILES = ["notes", "budget", "photo"]
WORDS = ["vide", "rapport", "facture", "todo", "resume"]   # index 0 = empty

# goal_key -> language intention, short menu description, and the concrete task
# spec the success check uses. spec[0] in {save, delete, mute, theme}.
GOALS = {
    "ecrire_rapport": ("Je dois ecrire un rapport et l'enregistrer dans mes notes.",
                       "ecrire 'rapport' dans le fichier notes",
                       ("save", 0, 1)),          # file notes, word rapport
    "ecrire_todo":    ("Note ma liste de taches dans le budget.",
                       "ecrire 'todo' dans le fichier budget",
                       ("save", 1, 3)),          # file budget, word todo
    "supprimer_photo": ("Supprime la photo, je n'en veux plus.",
                        "supprimer le fichier photo",
                        ("delete", 2)),          # file photo
    "muet":           ("Coupe le son.",
                       "mettre le volume en muet",
                       ("mute",)),
    "theme_sombre":   ("Passe en mode sombre.",
                       "activer le theme sombre",
                       ("theme",)),
}


class DesktopWorld:
    K = len(WORDS) - 1            # number of typeable words (excluding "vide")
    F = len(FILES)

    def __init__(self, max_steps: int = 12, step_cost: float = 0.02, seed: int = 0):
        self.max_steps = int(max_steps)
        self.step_cost = float(step_cost)
        # actions: 0 focus_fichiers, 1 focus_editeur, 2 focus_reglages,
        # 3 select_next, 4..4+K-1 type(word), save, delete, toggle
        self.action_space = 7 + self.K
        self._keys = list(GOALS.keys())
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def obs_dim(self) -> int:
        K1 = self.K + 1
        # focus + editor buffer + file presence + file content + vol + theme + sel + time
        return 3 + K1 + self.F + self.F * K1 + 1 + 1 + self.F + 1

    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.focus = 0
        self.buffer = 0                          # editor content (word index)
        self.selected = 0
        self.files_present = [0, 0, 1]           # notes/budget absent, photo present
        self.files_content = [0, 0, 0]
        self.volume_muted = 0
        self.theme_dark = 0
        self.goal_key = self._keys[int(self.rng.integers(0, len(self._keys)))]
        self.goal_spec = GOALS[self.goal_key][2]
        return self._obs()

    def _obs(self) -> np.ndarray:
        K1 = self.K + 1
        v = np.zeros(self.obs_dim, dtype=np.float32)
        o = 0
        v[o + self.focus] = 1.0;            o += 3
        v[o + self.buffer] = 1.0;           o += K1
        for i in range(self.F):
            v[o + i] = float(self.files_present[i])
        o += self.F
        for i in range(self.F):
            v[o + i * K1 + self.files_content[i]] = 1.0
        o += self.F * K1
        v[o] = float(self.volume_muted);    o += 1
        v[o] = float(self.theme_dark);      o += 1
        v[o + self.selected] = 1.0;         o += self.F
        v[o] = self.t / self.max_steps
        return v

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        a = int(action)
        if a == 0:
            self.focus = 0
        elif a == 1:
            self.focus = 1
        elif a == 2:
            self.focus = 2
        elif a == 3:
            self.selected = (self.selected + 1) % self.F
        elif 4 <= a <= 3 + self.K:                 # type a word (editor only)
            if self.focus == 1:
                self.buffer = a - 3
        elif a == 4 + self.K:                       # save buffer -> selected file
            if self.buffer != 0:
                self.files_present[self.selected] = 1
                self.files_content[self.selected] = self.buffer
        elif a == 5 + self.K:                       # delete selected file (files app)
            if self.focus == 0:
                self.files_present[self.selected] = 0
                self.files_content[self.selected] = 0
        elif a == 6 + self.K:                       # toggle setting (settings app)
            if self.focus == 2:
                if self.selected % 2 == 0:
                    self.volume_muted ^= 1
                else:
                    self.theme_dark ^= 1

        self.t += 1
        prog = self._progress()
        reached = self._achieved()
        reward = prog - self.step_cost
        done = reached or self.t >= self.max_steps
        return self._obs(), float(reward), bool(done), {"reached": reached}

    # --- goal / reward (depend on the hidden language intention) --- #
    def _progress(self) -> float:
        kind = self.goal_spec[0]
        if kind == "save":
            f, w = self.goal_spec[1], self.goal_spec[2]
            if self.files_present[f] and self.files_content[f] == w:
                return 1.0
            p = 0.0
            if self.selected == f:
                p += 0.2
            if self.buffer == w:
                p += 0.4
            return p
        if kind == "delete":
            f = self.goal_spec[1]
            if not self.files_present[f]:
                return 1.0
            return 0.3 if (self.focus == 0 and self.selected == f) else 0.0
        if kind == "mute":
            if self.volume_muted:
                return 1.0
            return 0.3 if (self.focus == 2 and self.selected % 2 == 0) else 0.0
        if kind == "theme":
            if self.theme_dark:
                return 1.0
            return 0.3 if (self.focus == 2 and self.selected % 2 == 1) else 0.0
        return 0.0

    def _achieved(self) -> bool:
        kind = self.goal_spec[0]
        if kind == "save":
            f, w = self.goal_spec[1], self.goal_spec[2]
            return bool(self.files_present[f] and self.files_content[f] == w)
        if kind == "delete":
            return not bool(self.files_present[self.goal_spec[1]])
        if kind == "mute":
            return bool(self.volume_muted)
        if kind == "theme":
            return bool(self.theme_dark)
        return False

    # --- language interface (shared vocab between state and goal targets) --- #
    def _format(self, focus, buffer, fp, fc, vol, theme, sel) -> str:
        def fdesc(i):
            if not fp[i]:
                return f"Fichier {FILES[i]} absent"
            return f"Fichier {FILES[i]} present {WORDS[fc[i]]}"
        return (f"Bureau. App {APPS[focus]}. Texte {WORDS[buffer]}. "
                f"{fdesc(0)}. {fdesc(1)}. {fdesc(2)}. "
                f"Volume {'muet' if vol else 'actif'}. "
                f"Theme {'sombre' if theme else 'clair'}. "
                f"Selection {FILES[sel]}.")

    def text_state(self) -> str:
        return self._format(self.focus, self.buffer, self.files_present,
                            self.files_content, self.volume_muted, self.theme_dark,
                            self.selected)

    def goal_instruction(self) -> str:
        return GOALS[self.goal_key][0]

    def true_goal_key(self):
        return self.goal_key

    def goal_menu(self):
        return [(k, v[1]) for k, v in GOALS.items()]

    def goal_state_text(self, key=None) -> str:
        """Canonical success state for a goal, in the SAME vocab as text_state()
        so the bridge maps it in-distribution (other fields at reset defaults)."""
        key = key or self.goal_key
        spec = GOALS[key][2]
        fp, fc = [0, 0, 1], [0, 0, 0]            # reset defaults
        focus, buf, vol, theme, sel = 0, 0, 0, 0, 0
        kind = spec[0]
        if kind == "save":
            f, w = spec[1], spec[2]
            fp[f], fc[f] = 1, w
            focus, buf, sel = 1, w, f
        elif kind == "delete":
            f = spec[1]
            fp[f], fc[f] = 0, 0
            focus, sel = 0, f
        elif kind == "mute":
            focus, vol, sel = 2, 1, 0
        elif kind == "theme":
            focus, theme, sel = 2, 1, 1
        return self._format(focus, buf, fp, fc, vol, theme, sel)
