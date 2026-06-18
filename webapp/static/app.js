const $ = (s) => document.querySelector(s);
const api = (p, o) => fetch(p, o).then(r => r.json());

const GOAL_LABELS = {
  productive: "Journée productive",
  balanced: "Journée équilibrée",
  recovery: "Récupération / repos",
};

async function boot() {
  try {
    const st = await api("/api/state");
    renderTasks(st.tasks);
    renderHabits(st.habits, []);
    renderChips(st.goals);
    $("#status").classList.add("ok");
    $("#status-text").textContent = "modèle prêt";
  } catch (e) {
    $("#status-text").textContent = "backend hors ligne";
  }
  loadLab();
}

function renderChips(goals) {
  const c = $("#chips");
  c.innerHTML = "";
  goals.forEach(g => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = g.intention;
    b.onclick = () => { $("#intention").value = g.intention; };
    c.appendChild(b);
  });
}

function renderTasks(tasks) {
  $("#tasks").innerHTML = tasks.map(t => `
    <li><span class="check"></span>
      <span>Tâche</span>
      <span class="badge ${t.priority === 'haute' ? 'high' : ''}">${t.priority}</span>
    </li>`).join("");
}

function renderHabits(habits, done) {
  const fr = { exercise: "Sport", study: "Étude", social: "Social" };
  $("#habits").innerHTML = habits.map(h => `
    <li><span class="check ${done.includes(h) ? 'on' : ''}"></span>${fr[h] || h}</li>
  `).join("");
}

async function loadLab() {
  const lab = await api("/api/lab");
  const el = $("#lab");
  if (!lab.summary) { el.innerHTML = '<p class="muted small">Lance d\'abord la recherche : <code>python loop/random_search.py</code></p>'; return; }
  const s = lab.summary;
  el.innerHTML = `
    <div class="kpi">
      <div class="box"><b>${fmt(s.bridge_gain)}</b><span>gain · pont ON (${s.n_bridge})</span></div>
      <div class="box"><b>${fmt(s.control_gain)}</b><span>gain · contrôle (${s.n_control})</span></div>
    </div>
    ${lab.rows.map(r => `<div class="labrow"><span>${r.tag}</span><span>score ${r.score} · gain ${fmt(r.gain)}</span></div>`).join("")}
  `;
}
const fmt = (x) => x === null || x === undefined ? "—" : (x >= 0 ? "+" : "") + x;

async function plan() {
  const btn = $("#plan-btn");
  btn.disabled = true; btn.textContent = "Planification…";
  try {
    const res = await api("/api/plan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ intention: $("#intention").value }),
    });
    renderPlan(res);
  } catch (e) {
    alert("Erreur de planification : " + e);
  }
  btn.disabled = false; btn.textContent = "Planifier ma journée";
}

function renderPlan(res) {
  $("#result").classList.remove("hidden");
  $("#goal-title").textContent = GOAL_LABELS[res.goal_key] || res.goal_key;
  $("#goal-desc").textContent = res.goal_desc;

  $("#outcome").innerHTML = `
    <div class="pill ${res.achieved ? 'ok' : ''}"><b>${res.achieved ? "✓" : "—"}</b>objectif</div>
    <div class="pill"><b>${Math.round(res.score * 100)}%</b>score jour</div>
    <div class="pill"><b>${Math.round(res.final_energy * 100)}%</b>énergie</div>`;

  $("#timeline").innerHTML = res.timeline.map(s => `
    <div class="slot">
      <div class="n">${s.slot}</div>
      <div class="act">${s.label}</div>
      <div class="meta">
        ${s.habits_done.map(h => `<span class="tag h">${frHabit(h)}</span>`).join("")}
        ${s.high_done ? `<span class="tag">${s.high_done} tâche(s)</span>` : ""}
        <div class="ebar"><i style="width:${Math.round(s.energy * 100)}%"></i></div>
      </div>
    </div>`).join("");

  renderHabits(["exercise", "study", "social"], res.habits_done);
  drawEnergy(res.timeline.map(s => s.energy));
}
const frHabit = (h) => ({ exercise: "sport", study: "étude", social: "social" }[h] || h);

function drawEnergy(vals) {
  const svg = $("#energy");
  const W = 320, H = 120, pad = 8;
  const n = vals.length;
  const x = i => pad + (W - 2 * pad) * (n === 1 ? 0.5 : i / (n - 1));
  const y = v => H - pad - (H - 2 * pad) * Math.max(0, Math.min(1, v));
  const pts = vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const area = `${pad},${H - pad} ${pts} ${W - pad},${H - pad}`;
  svg.innerHTML = `
    <defs><linearGradient id="g" x1="0" x2="1">
      <stop offset="0" stop-color="#7c9cff"/><stop offset="1" stop-color="#5eead4"/>
    </linearGradient></defs>
    <polygon class="earea" points="${area}"/>
    <polyline class="eline" points="${pts}"/>`;
  $("#energy-cap").textContent = `de ${Math.round(vals[0] * 100)}% à ${Math.round(vals[vals.length - 1] * 100)}% sur la journée`;
}

$("#plan-btn").addEventListener("click", plan);
boot();
