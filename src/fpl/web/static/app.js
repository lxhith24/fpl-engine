"use strict";

const params = new URLSearchParams(location.search);
const ENTRY = params.get("entry") ? Number(params.get("entry")) : undefined;

const state = {
  snap: null,
  chat: [],          // {role, content} committed turns
  streaming: false,
  predSort: { key: "xp", dir: "desc" },
  predFilter: { name: "", pos: "" },
};

const $ = (sel) => document.querySelector(sel);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const fmt = {
  xp: (v) => (v == null ? "–" : v.toFixed(2)),
  cost: (v) => (v == null ? "–" : "£" + v.toFixed(1) + "m"),
  pct: (v) => (v == null ? "–" : v.toFixed(1) + "%"),
  p2: (v) => (v == null ? "–" : v.toFixed(2)),
  int: (v) => (v == null ? "–" : Math.round(v).toString()),
  zone: (v) => (v == null ? "–" : v.toFixed(2) + "×"),
};

function pos(p) {
  return `<span class="pill pos-${esc(p)}">${esc(p)}</span>`;
}
function oppCell(p) {
  if (!p.opponent) return "–";
  return `${esc(p.opponent)} <span class="muted">(${p.is_home ? "H" : "A"})</span>`;
}

// ------------------------------------------------------------------ snapshot
async function loadSnapshot(refresh) {
  const qs = new URLSearchParams();
  if (ENTRY) qs.set("entry", ENTRY);
  if (refresh) qs.set("refresh", "1");
  $("#main").innerHTML = `<div class="card"><span class="muted">Loading snapshot…</span></div>`;
  const res = await fetch("/api/snapshot?" + qs.toString());
  const snap = await res.json();
  state.snap = snap;
  render();
}

function render() {
  const s = state.snap;
  $("#gw").textContent = "GW" + (s.gw ?? "–");
  $("#meta").textContent = [
    s.generated_at ? "generated " + s.generated_at.replace("T", " ").slice(0, 16) : null,
    "entry " + s.entry_id,
  ]
    .filter(Boolean)
    .join("  ·  ");

  const banner = $("#banner");
  if (s.chat_configured === false) {
    banner.hidden = false;
    banner.textContent =
      "Chat is disabled — set ANTHROPIC_API_KEY in the environment and restart `python -m fpl web`.";
  } else {
    banner.hidden = true;
  }

  if (!s.has_predictions) {
    $("#main").innerHTML = `<div class="card"><h2>No predictions</h2>
      <p class="note">${esc((s.notes || []).join("; ") || "Run `python -m fpl predict` first.")}</p></div>`;
    renderSuggests();
    return;
  }

  const parts = [];
  parts.push(optimalCard(s.optimal));
  parts.push(yourSquadCard(s.your_squad, s.optimal));
  if (s.differentials?.length) parts.push(differentialsCard(s.differentials));
  if (s.matchup_boosts?.length) parts.push(boostsCard(s.matchup_boosts));
  parts.push(predictionsCard(s.predictions || []));
  if (s.report_md)
    parts.push(
      `<details class="report"><summary>Full GW${s.gw} briefing (markdown)</summary><pre>${esc(
        s.report_md
      )}</pre></details>`
    );
  if (s.notes?.length)
    parts.push(`<p class="note">${esc(s.notes.join(" · "))}</p>`);
  $("#main").innerHTML = parts.join("");

  wirePredictionsControls();
  renderSuggests();
}

function playerRows(list, cols) {
  return list
    .map((p) => {
      const tds = cols.map((c) => c(p)).join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");
}

function optimalCard(o) {
  if (!o) return "";
  const maxXp = Math.max(...o.xi.map((p) => p.xp || 0), 1);
  const rows = o.xi
    .map((p) => {
      const isC = p.web_name === o.captain.web_name;
      const isV = p.web_name === o.vice.web_name;
      const tag = isC ? ' <span class="pill">C</span>' : isV ? ' <span class="pill">V</span>' : "";
      return `<tr>
        <td>${esc(p.web_name)}${tag}<div class="bar"><i style="width:${((p.xp || 0) / maxXp) * 100}%"></i></div></td>
        <td>${pos(p.position)}</td>
        <td>${esc(p.team)}</td>
        <td>${oppCell(p)}</td>
        <td class="num">${fmt.cost(p.now_cost)}</td>
        <td class="num">${fmt.pct(p.selected_by_percent)}</td>
        <td class="num">${fmt.p2(p.p_start)}</td>
        <td class="num">${fmt.xp(p.xp_sigma)}</td>
        <td class="num">${fmt.xp(p.xp)}</td>
      </tr>`;
    })
    .join("");
  return `<div class="card">
    <h2>Optimal XI · risk aggressive</h2>
    <div class="stats">
      <div class="stat"><div class="v big good">${fmt.xp(o.xi_xp)}</div><div class="k">XI xP</div></div>
      <div class="stat"><div class="v">${esc(o.formation)}</div><div class="k">formation</div></div>
      <div class="stat"><div class="v">${fmt.cost(o.total_cost)}</div><div class="k">cost</div></div>
      <div class="stat"><div class="v">${fmt.pct(o.mean_ownership)}</div><div class="k">mean own</div></div>
      <div class="stat"><div class="v">${esc(o.captain.web_name)}</div><div class="k">captain · ${fmt.xp(
        o.captain.xp
      )}</div></div>
      <div class="stat"><div class="v">${esc(o.vice.web_name)}</div><div class="k">vice · ${fmt.xp(
        o.vice.xp
      )}</div></div>
    </div>
    <div class="scroll"><table>
      <thead><tr><th>Player</th><th>Pos</th><th>Team</th><th>Opp</th>
        <th class="num">Price</th><th class="num">Own</th><th class="num">p̂ start</th>
        <th class="num">σ</th><th class="num">xP</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <div class="bench">Bench: ${o.bench
      .map((p) => `${esc(p.web_name)} (${fmt.xp(p.xp)})`)
      .join(" · ")}</div>
  </div>`;
}

function yourSquadCard(y, o) {
  if (!y || y.available === false) {
    return `<div class="card"><h2>Your squad</h2>
      <p class="note">${esc(y?.reason || "Squad unavailable — could not fetch manager picks.")}</p></div>`;
  }
  const t = y.transfer;
  const transferHtml = t
    ? `<div class="transfer">
        <span class="out">OUT ${esc(t.out.web_name)} ${pos(t.out.position)} ${fmt.cost(
        t.out.now_cost
      )} · ${fmt.xp(t.out.xp)} xP</span>
        <span class="arrow">→</span>
        <span class="in">IN ${esc(t.in.web_name)} ${pos(t.in.position)} ${fmt.cost(
        t.in.now_cost
      )} · ${fmt.xp(t.in.xp)} xP</span>
        <span class="muted">+${fmt.xp(t.gain)} xP, net ${t.net_cost >= 0 ? "+" : ""}${t.net_cost.toFixed(
        1
      )}m</span>
      </div>`
    : `<p class="muted">No positive single transfer found.</p>`;
  const rows = playerRows(y.xi, [
    (p) => `<td>${esc(p.web_name)}</td>`,
    (p) => `<td>${pos(p.position)}</td>`,
    (p) => `<td>${esc(p.team)}</td>`,
    (p) => `<td>${oppCell(p)}</td>`,
    (p) => `<td class="num">${fmt.cost(p.now_cost)}</td>`,
    (p) => `<td class="num">${fmt.pct(p.selected_by_percent)}</td>`,
    (p) => `<td class="num">${fmt.xp(p.xp)}</td>`,
  ]);
  return `<div class="card">
    <h2>Your squad · entry ${esc(state.snap.entry_id)}</h2>
    <div class="stats">
      <div class="stat"><div class="v big">${fmt.xp(y.xi_xp)}</div><div class="k">best legal XI xP</div></div>
      <div class="stat"><div class="v warn">+${fmt.xp(y.gap)}</div><div class="k">gap to optimum</div></div>
      <div class="stat"><div class="v">${y.n_transfers ?? "–"}</div><div class="k">transfers to optimum</div></div>
      <div class="stat"><div class="v">−${y.hit_cost ?? 0}</div><div class="k">hit cost (pts)</div></div>
      <div class="stat"><div class="v">${esc(y.formation)}</div><div class="k">formation</div></div>
    </div>
    <strong class="muted">Best single transfer</strong>
    ${transferHtml}
    <div class="scroll" style="margin-top:12px"><table>
      <thead><tr><th>Player</th><th>Pos</th><th>Team</th><th>Opp</th>
        <th class="num">Price</th><th class="num">Own</th><th class="num">xP</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </div>`;
}

function differentialsCard(list) {
  const rows = list
    .map(
      (d) => `<tr><td>${esc(d.web_name)}</td><td>${pos(d.position)}</td><td>${esc(
        d.team
      )}</td><td class="num">${fmt.pct(d.selected_by_percent)}</td><td class="num">${fmt.xp(
        d.xp
      )}</td></tr>`
    )
    .join("");
  return `<div class="card"><h2>Differentials · &lt;5% owned, xP &gt; 4</h2>
    <div class="scroll"><table>
      <thead><tr><th>Player</th><th>Pos</th><th>Team</th><th class="num">Own</th><th class="num">xP</th></tr></thead>
      <tbody>${rows}</tbody></table></div></div>`;
}

function boostsCard(list) {
  const rows = list
    .map(
      (b) =>
        `<li><strong>${esc(b.web_name)}</strong> <span class="muted">${esc(b.team)}${
          b.opponent ? " vs " + esc(b.opponent) : ""
        }</span> — ${fmt.zone(b.zone_fit)} into weak zones, ${fmt.xp(b.xp)} xP</li>`
    )
    .join("");
  return `<div class="card"><h2>Matchup boosts · zone fit</h2><ul>${rows}</ul></div>`;
}

// ------------------------------------------------------ predictions explorer
const PRED_COLS = [
  { key: "web_name", label: "Player", cell: (p) => esc(p.web_name) },
  { key: "team", label: "Team", cell: (p) => esc(p.team) },
  { key: "position", label: "Pos", cell: (p) => pos(p.position), num: false },
  { key: "opponent", label: "Opp", cell: oppCell },
  { key: "now_cost", label: "Price", num: true, cell: (p) => fmt.cost(p.now_cost) },
  { key: "selected_by_percent", label: "Own", num: true, cell: (p) => fmt.pct(p.selected_by_percent) },
  { key: "form", label: "Form", num: true, cell: (p) => fmt.p2(p.form) },
  { key: "p_start", label: "p̂ start", num: true, cell: (p) => fmt.p2(p.p_start) },
  { key: "exp_minutes", label: "x·min", num: true, cell: (p) => fmt.int(p.exp_minutes) },
  { key: "zone_fit", label: "Zone", num: true, cell: (p) => fmt.zone(p.zone_fit) },
  { key: "xp_sigma", label: "σ", num: true, cell: (p) => fmt.xp(p.xp_sigma) },
  { key: "xp", label: "xP", num: true, cell: (p) => fmt.xp(p.xp) },
];

function predictionsCard(list) {
  return `<div class="card">
    <h2>Predictions explorer · <span id="pred-count" class="muted">${list.length}</span></h2>
    <div class="controls">
      <input id="pred-name" type="search" placeholder="filter by name…" value="${esc(state.predFilter.name)}" />
      <select id="pred-pos">
        <option value="">all positions</option>
        ${["GKP", "DEF", "MID", "FWD"]
          .map((p) => `<option value="${p}" ${state.predFilter.pos === p ? "selected" : ""}>${p}</option>`)
          .join("")}
      </select>
    </div>
    <div class="scroll tall"><table id="pred-table">
      <thead><tr>${PRED_COLS.map(
        (c) =>
          `<th data-key="${c.key}" class="${c.num ? "num " : ""}${
            state.predSort.key === c.key ? "sort-" + state.predSort.dir : ""
          }">${c.label}</th>`
      ).join("")}</tr></thead>
      <tbody>${predBody(list)}</tbody>
    </table></div>
  </div>`;
}

function predBody(list) {
  const { name, pos: p } = state.predFilter;
  const nlc = name.toLowerCase();
  let rows = list.filter(
    (r) => (!nlc || (r.web_name || "").toLowerCase().includes(nlc)) && (!p || r.position === p)
  );
  const { key, dir } = state.predSort;
  rows.sort((a, b) => {
    const av = a[key], bv = b[key];
    if (av == null) return 1;
    if (bv == null) return -1;
    const cmp = typeof av === "number" ? av - bv : String(av).localeCompare(String(bv));
    return dir === "asc" ? cmp : -cmp;
  });
  const count = $("#pred-count");
  if (count) count.textContent = rows.length;
  return rows
    .map((r) => `<tr>${PRED_COLS.map((c) => `<td class="${c.num ? "num" : ""}">${c.cell(r)}</td>`).join("")}</tr>`)
    .join("");
}

function wirePredictionsControls() {
  const list = state.snap.predictions || [];
  const redraw = () => {
    $("#pred-table tbody").innerHTML = predBody(list);
  };
  $("#pred-name")?.addEventListener("input", (e) => {
    state.predFilter.name = e.target.value;
    redraw();
  });
  $("#pred-pos")?.addEventListener("change", (e) => {
    state.predFilter.pos = e.target.value;
    redraw();
  });
  $("#pred-table")
    ?.querySelectorAll("th")
    .forEach((th) =>
      th.addEventListener("click", () => {
        const key = th.dataset.key;
        if (state.predSort.key === key) {
          state.predSort.dir = state.predSort.dir === "asc" ? "desc" : "asc";
        } else {
          state.predSort.key = key;
          state.predSort.dir = key === "web_name" || key === "team" ? "asc" : "desc";
        }
        render();
      })
    );
}

// ------------------------------------------------------------------- chat
function mdInline(s) {
  return esc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderSuggests() {
  const box = $("#suggests");
  const s = state.snap;
  if (!box) return;
  if (!s?.has_predictions || s.chat_configured === false || state.chat.length) {
    box.innerHTML = "";
    return;
  }
  const cap = s.optimal?.captain?.web_name;
  const tr = s.your_squad?.transfer;
  const qs = [
    cap && `Why is ${cap} the captain pick?`,
    tr && `Make the case for ${tr.in.web_name} over ${tr.out.web_name}.`,
    "Which differentials have the highest ceiling?",
    s.your_squad?.available && "How far is my squad from optimal, and why?",
  ].filter(Boolean);
  box.innerHTML = qs.map((q) => `<button>${esc(q)}</button>`).join("");
  box.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      $("#chat-text").value = b.textContent;
      sendChat();
    })
  );
}

function addBubble(role, html) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.innerHTML = html;
  $("#chat-log").appendChild(div);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return div;
}

async function sendChat() {
  if (state.streaming) return;
  const ta = $("#chat-text");
  const text = ta.value.trim();
  if (!text) return;
  if (!state.snap?.has_predictions) {
    addBubble("error", "No gameweek data loaded to talk about.");
    return;
  }
  ta.value = "";
  ta.style.height = "38px";
  state.chat.push({ role: "user", content: text });
  addBubble("user", esc(text));
  renderSuggests();

  const bubble = addBubble("assistant", '<span class="cursor">▍</span>');
  state.streaming = true;
  $("#chat-send").disabled = true;
  let acc = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: state.chat, entry: state.snap.entry_id }),
    });
    if (!res.ok || !res.body) throw new Error("HTTP " + res.status);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (!line.startsWith("data: ")) continue;
        const obj = JSON.parse(line.slice(6));
        if (obj.delta) {
          acc += obj.delta;
          bubble.innerHTML = mdInline(acc) + '<span class="cursor">▍</span>';
          $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
        } else if (obj.error) {
          bubble.className = "msg error";
          bubble.textContent = obj.error;
          acc = "";
        }
      }
    }
  } catch (e) {
    bubble.className = "msg error";
    bubble.textContent = "Request failed: " + e.message;
    acc = "";
  } finally {
    state.streaming = false;
    $("#chat-send").disabled = false;
  }

  if (acc) {
    bubble.innerHTML = mdInline(acc);
    state.chat.push({ role: "assistant", content: acc });
  } else if (bubble.className === "msg assistant") {
    bubble.remove();
  }
}

// ------------------------------------------------------------------- wiring
$("#refresh").addEventListener("click", () => loadSnapshot(true));
$("#chat-send").addEventListener("click", sendChat);
$("#chat-clear").addEventListener("click", () => {
  state.chat = [];
  $("#chat-log").innerHTML = "";
  renderSuggests();
});
$("#chat-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
});
$("#chat-text").addEventListener("input", (e) => {
  e.target.style.height = "38px";
  e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
});

loadSnapshot(false);
