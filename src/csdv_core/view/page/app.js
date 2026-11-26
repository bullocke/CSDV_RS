/* CSDV map viewer: the pair-model disturbance maps over NAIP, two runs side
   by side, with comment points that save by themselves.

   All state is in S. Every change of state calls refresh(), which sets the
   controls and the map layers from it and writes it to the URL hash. */
"use strict";

const $ = (id) => document.getElementById(id);
const PANELS = ["A", "B"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const TAG_LABELS = {
  false_positive: "False positive",
  missed: "Missed",
  wrong_type: "Wrong type",
  wrong_date: "Wrong date",
  good: "Good",
  other: "Other",
};
const TAG_COLOURS = {
  false_positive: "#d1495b",
  missed: "#e8a22b",
  wrong_type: "#8e5fc2",
  wrong_date: "#00798c",
  good: "#3a9d5d",
  other: "#6b6a66",
};
const AGENT_CODE = { harvest: 1, wind: 2, fire: 3, biotic_mortality: 4, conversion: 5, other: 6 };
const AGENT_SHORT = ["", "harvest", "wind", "fire", "insects", "conversion", "other"];
const FAMILY_SHORT = {
  look: "non-holdout",
  holdout: "holdout",
  holdout_sf0: "sf 0%",
  holdout_sf: "sf 15%",
  di: "DI",
};
/* Background year layers kept loaded per panel, so going back is instant. */
const MAX_BG = 2;
/* The 0.6 m imagery replaces the local 3 m quicklook from this zoom. */
const HIRES_MIN_ZOOM = 15;
const SAVE_DELAY_MS = 500;

let CAT = null;
const SITES = {};
const S = {
  site: null,
  run: { A: null, B: null },
  thr: { A: null, B: null }, // percent, null for the threshold of the run
  layer: "date",
  interval: "all",
  year: null,
  bg: "naip",
  opacity: 0.85,
  overlay: true,
  labels: false,
  dual: true,
  forest: false,
  hires: true,
};
const maps = {};
const over = {};
const bg = { A: new Map(), B: new Map() };
const labelGroup = {};
const noteGroup = {};
const cursorMark = {};
const pickMark = {};
const siteExtra = {}; // unit -> answer of /api/site
const labelData = {}; // unit -> GeoJSON
const notes = new Map(); // id -> feature
let sel = null; // the open point: {id, lat, lon, isNew, dirty, props}
let point = null; // answer of /api/point for the open point
let lastLatLng = null;
let saveTimer = null;
let peek = false;
let tagFilter = "";
let syncing = false;

/* ---------- small helpers ---------- */

const site = () => SITES[S.site];
const run = (k) => CAT.runs[S.run[k]];
const panels = () => (S.dual ? PANELS : ["A"]);
const intervalYears = (iv) => [Number(iv.slice(0, 4)), Number(iv.slice(11, 15))];
const intervalLabel = (iv) => `${iv.slice(2, 4)}–${iv.slice(13, 15)}`;
const esc = (text) =>
  String(text ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function defaultThr(r) {
  return r.family === "di" ? 0 : Math.round(r.threshold * 100);
}

function thr(k) {
  const r = run(k);
  return r.family === "di" ? 0 : (S.thr[k] ?? defaultThr(r));
}

function siteBounds(pad) {
  const b = site().bbox;
  if (!b) return undefined;
  return L.latLngBounds([b[1], b[0]], [b[3], b[2]]).pad(pad);
}

function agentColour(code) {
  const hit = CAT.legend.agents.find((a) => a.code === code);
  return hit ? hit.colour : "#ffffff";
}

function runShort(r) {
  return FAMILY_SHORT[r.family] || r.run;
}

/* ---------- maps ---------- */

function makeMap(k) {
  const map = L.map("map" + k, {
    zoomAnimation: false,
    fadeAnimation: false,
    attributionControl: false,
    zoomControl: k === "A",
    keyboard: false,
    doubleClickZoom: false,
    minZoom: 9,
    maxZoom: 20,
  });
  for (const [name, z] of [["bg", 200], ["over", 300], ["labels", 350], ["notes", 450], ["cursor", 460]]) {
    map.createPane(name).style.zIndex = z;
  }
  map.getPane("labels").style.pointerEvents = "none";
  map.getPane("cursor").style.pointerEvents = "none";
  L.control.scale({ imperial: false }).addTo(map);
  labelGroup[k] = L.layerGroup().addTo(map);
  noteGroup[k] = L.layerGroup().addTo(map);
  const other = k === "A" ? "B" : "A";
  map.on("move zoomend", () => {
    if (syncing || !S.dual) return;
    syncing = true;
    maps[other].setView(map.getCenter(), map.getZoom(), { animate: false });
    syncing = false;
  });
  map.on("moveend", () => {
    writeHash();
    showStatus();
  });
  map.on("mousemove", (e) => {
    lastLatLng = e.latlng;
    if (!S.dual) return;
    if (!cursorMark[other]) {
      cursorMark[other] = L.circleMarker(e.latlng, {
        pane: "cursor", radius: 5, color: "#ffffff", weight: 1.5, fill: false, interactive: false,
      });
    }
    cursorMark[other].setLatLng(e.latlng).addTo(maps[other]);
  });
  map.on("mouseout", () => {
    if (cursorMark[other]) cursorMark[other].remove();
  });
  map.on("click", (e) => selectPoint(e.latlng.lat, e.latlng.lng));
  maps[k] = map;
}

function overlayUrl(k) {
  const r = run(k);
  if (!r || !r.layers.includes(S.layer)) return null;
  if (S.interval !== "all" && !r.intervals.includes(S.interval)) return null;
  const mask = S.forest ? "forest" : "all";
  return `/tiles/map/${r.run}/${S.layer}/${S.interval}/${thr(k)}/${mask}/{z}/{x}/{y}.png?v=${r.version}`;
}

function updateOverlay(k) {
  const url = panels().includes(k) ? overlayUrl(k) : null;
  if (!url) {
    if (over[k]) maps[k].removeLayer(over[k]);
    over[k] = null;
    return;
  }
  if (!over[k]) {
    over[k] = L.tileLayer(url, {
      pane: "over", maxNativeZoom: 17, maxZoom: 20, bounds: siteBounds(0.02), keepBuffer: 3,
    }).addTo(maps[k]);
  } else if (over[k]._url !== url) {
    over[k].setUrl(url);
  }
  over[k].setOpacity(S.overlay && !peek ? S.opacity : 0);
}

function previousYear(years, year) {
  const before = years.filter((y) => y < year);
  return before.length ? before[before.length - 1] : null;
}

/* What the background shows now: a cache key, the tile layers and a note. */
function bgSpec() {
  const s = site();
  const y = S.year;
  if (S.bg === "chm") {
    if (!s.chm_years.includes(y)) return { key: `chm:${y}`, layers: [], note: `No canopy height for ${y}` };
    return {
      key: `chm:${y}`,
      layers: [{ url: `/tiles/chm/${s.unit}/${y}/{z}/{x}/{y}.png`, maxNativeZoom: 16 }],
      note: `Canopy height ${y}`,
    };
  }
  if (S.bg === "dchm") {
    const y0 = previousYear(s.chm_years, y);
    if (y0 === null || !s.chm_years.includes(y)) {
      return { key: `dchm:${y}`, layers: [], note: `No height change ending in ${y}` };
    }
    return {
      key: `dchm:${y0}:${y}`,
      layers: [{ url: `/tiles/dchm/${s.unit}/${y0}/${y}/{z}/{x}/{y}.png`, maxNativeZoom: 16 }],
      note: `Height change ${y0} to ${y} (red loss, blue gain)`,
    };
  }
  const layers = [];
  const local = s.quicklook_years.includes(y);
  if (local) layers.push({ url: `/tiles/naip/${s.unit}/${y}/{z}/{x}/{y}`, maxNativeZoom: 16 });
  const remote = S.hires && siteExtra[s.unit] && siteExtra[s.unit].naip[y];
  if (remote) {
    layers.push({
      url: remote.url,
      maxNativeZoom: remote.max_native_zoom,
      minZoom: local ? HIRES_MIN_ZOOM : 0,
    });
  }
  let note = "";
  if (!layers.length) note = `No NAIP for ${y}`;
  else if (remote && remote.kind === "usda") note = "0.6 m from USDA: its newest collection, which may be a later year";
  else if (!remote && S.hires && siteExtra[s.unit]) note = "3 m quicklook only for this year";
  return { key: `naip:${y}:${remote ? 1 : 0}`, layers, note };
}

function clearBg(k) {
  for (const layers of bg[k].values()) layers.forEach((l) => maps[k].removeLayer(l));
  bg[k].clear();
}

function updateBg(k) {
  if (!panels().includes(k)) {
    clearBg(k);
    return;
  }
  const cache = bg[k];
  const spec = bgSpec();
  let layers = cache.get(spec.key);
  if (layers) {
    cache.delete(spec.key); // put it back as the newest entry
  } else {
    layers = spec.layers.map((d, i) =>
      L.tileLayer(d.url, {
        pane: "bg",
        maxNativeZoom: d.maxNativeZoom,
        maxZoom: 20,
        minZoom: d.minZoom || 0,
        zIndex: i + 1,
        bounds: siteBounds(0.25),
        keepBuffer: 3,
      }).addTo(maps[k]),
    );
  }
  cache.set(spec.key, layers);
  for (const [key, held] of cache) held.forEach((l) => l.setOpacity(key === spec.key ? 1 : 0));
  while (cache.size > MAX_BG) {
    const oldest = cache.keys().next().value;
    cache.get(oldest).forEach((l) => maps[k].removeLayer(l));
    cache.delete(oldest);
  }
}

function showStatus() {
  const spec = bgSpec();
  const zoom = maps.A.getZoom();
  const parts = [spec.note];
  if (S.bg === "naip" && S.hires && zoom < HIRES_MIN_ZOOM && spec.layers.length > 1) {
    parts.push(`0.6 m from zoom ${HIRES_MIN_ZOOM}`);
  }
  parts.push(`zoom ${zoom}`);
  $("status").textContent = parts.filter(Boolean).join(" · ");
}

/* ---------- labels ---------- */

function yearPairs(text) {
  const pairs = [];
  for (const m of String(text || "").matchAll(/(\d{4})-\d\d-\d\d_(\d{4})-\d\d-\d\d/g)) pairs.push(`${m[1]}_${m[2]}`);
  return pairs;
}

async function loadLabels() {
  const unit = S.site;
  if (!labelData[unit]) {
    try {
      const res = await fetch(`/api/labels/${unit}`);
      labelData[unit] = res.ok ? await res.json() : { type: "FeatureCollection", features: [] };
    } catch (err) {
      labelData[unit] = { type: "FeatureCollection", features: [] };
    }
  }
  if (unit === S.site) drawLabels();
}

function drawLabels() {
  const data = labelData[S.site];
  const want = S.interval === "all" ? null : intervalYears(S.interval).join("_");
  for (const k of PANELS) {
    labelGroup[k].clearLayers();
    if (!S.labels || !data || !panels().includes(k)) continue;
    L.geoJSON(data, {
      pane: "labels",
      interactive: false,
      filter: (f) => {
        if (!want) return true;
        const p = f.properties;
        return yearPairs(p.layer === "events" ? p.loss_intervals : p.interval).includes(want);
      },
      style: (f) => {
        const p = f.properties;
        if (p.layer === "events") {
          return { color: agentColour(AGENT_CODE[p.agent]), weight: 2, fill: false };
        }
        return { color: "#ffffff", weight: 1, opacity: 0.6, dashArray: "4 4", fill: false };
      },
    }).addTo(labelGroup[k]);
  }
}

/* ---------- controls ---------- */

function segButton(text, on, title, click) {
  const b = document.createElement("button");
  b.innerHTML = text;
  if (on) b.classList.add("on");
  if (title) b.title = title;
  b.addEventListener("click", () => {
    click();
    b.blur();
  });
  return b;
}

function buildIntervals() {
  const box = $("intervals");
  box.replaceChildren(segButton("All", S.interval === "all", "Every interval", () => setInterval_("all")));
  for (const iv of site().intervals) {
    box.append(segButton(intervalLabel(iv), S.interval === iv, iv.replace("_", " to "), () => setInterval_(iv)));
  }
}

function buildYears() {
  const box = $("years");
  box.replaceChildren();
  const [before, after] = S.interval === "all" ? [null, null] : intervalYears(S.interval);
  for (const { year, date } of site().years) {
    const month = date ? ` <small>${MONTHS[Number(date.slice(5, 7)) - 1]}</small>` : "";
    const b = segButton(`${year}${month}`, S.year === year, date || String(year), () => {
      S.year = year;
      refresh();
    });
    if (year === before) b.classList.add("before");
    if (year === after) b.classList.add("after");
    box.append(b);
  }
}

function scoreText(r) {
  let scores = r.scores.model;
  if (r.family === "di") {
    const withDi = site().runs.map((name) => CAT.runs[name]).find((x) => x.scores.di);
    scores = withDi ? withDi.scores.di : null;
  }
  if (!scores || scores.average_precision === undefined) return "";
  const f = (v) => (v === undefined ? "–" : v.toFixed(2));
  return `AP ${f(scores.average_precision)}  P ${f(scores.precision)}  R ${f(scores.recall)}  F1 ${f(scores.f1)}`;
}

function updateHead(k) {
  const el = $("panel" + k);
  const r = run(k);
  const select = el.querySelector(".run");
  select.replaceChildren();
  for (const name of site().runs) {
    const x = CAT.runs[name];
    const o = new Option(x.family === "di" ? x.label : `${x.label} (${x.run})`, name);
    select.append(o);
  }
  select.value = r.run;
  const box = el.querySelector(".thr");
  const slider = box.querySelector("input");
  const fixed = r.family === "di" || S.layer === "prob";
  slider.disabled = fixed;
  box.style.visibility = fixed ? "hidden" : "visible";
  slider.value = thr(k);
  box.querySelector("output").textContent = (thr(k) / 100).toFixed(2);
  box.classList.toggle("changed", S.thr[k] !== null && S.thr[k] !== defaultThr(r));
  const scores = el.querySelector(".scores");
  let text = scoreText(r);
  if (!r.layers.includes(S.layer)) text = `No ${S.layer} layer for this source`;
  else if (S.interval !== "all" && !r.intervals.includes(S.interval)) text = "This run has no such interval";
  scores.textContent = text;
  scores.title = "Scores on the label chips of this unit, at the threshold of the run";
}

function buildLegend() {
  const box = $("legend");
  const chip = (colour, text) => `<span class="chip"><span class="sw" style="background:${colour}"></span>${esc(text)}</span>`;
  if (S.layer === "type") {
    box.innerHTML = CAT.legend.agents.map((a) => chip(a.colour, a.label)).join("");
  } else if (S.layer === "prob") {
    const ramp = CAT.legend.prob.map((c) => `<span style="background:${c}"></span>`).join("");
    box.innerHTML = `Loss probability ${CAT.legend.prob_min}% <span class="ramp">${ramp}</span> 100%`;
  } else {
    let years = [...new Set(site().intervals.map((iv) => intervalYears(iv)[1]))];
    if (S.interval !== "all") years = [intervalYears(S.interval)[1]];
    box.innerHTML = "Loss first seen by " + years.map((y) => chip(CAT.legend.years[y] || "#888", y)).join("");
  }
}

function refresh() {
  const s = site();
  for (const b of $("layer").children) b.classList.toggle("on", b.dataset.layer === S.layer);
  buildIntervals();
  buildYears();
  $("forest").checked = S.forest;
  $("labels").checked = S.labels;
  $("hires").checked = S.hires;
  $("opacity").value = Math.round(S.opacity * 100);
  $("bg").value = S.bg;
  $("dual").textContent = S.dual ? "One panel" : "Two panels";
  $("panelB").hidden = !S.dual;
  for (const k of PANELS) {
    updateHead(k);
    updateOverlay(k);
    updateBg(k);
  }
  buildLegend();
  drawLabels();
  showStatus();
  writeHash();
  if (sel) renderInspector();
  document.title = `${s.place} · CSDV map viewer`;
}

function setInterval_(iv) {
  S.interval = iv;
  if (iv !== "all") S.year = intervalYears(iv)[1];
  refresh();
}

function stepList(list, current, d) {
  const i = list.indexOf(current);
  return list[(i + d + list.length) % list.length];
}

function applySite(view) {
  const s = site();
  $("site").value = s.unit;
  PANELS.forEach((k, i) => {
    if (!s.runs.includes(S.run[k])) {
      S.run[k] = s.runs[Math.min(i, s.runs.length - 1)];
      S.thr[k] = null;
    }
  });
  if (S.interval !== "all" && !s.intervals.includes(S.interval)) S.interval = "all";
  const years = s.years.map((y) => y.year);
  if (!years.includes(S.year)) S.year = years[years.length - 1];
  for (const k of PANELS) {
    clearBg(k);
    if (over[k]) maps[k].removeLayer(over[k]);
    over[k] = null;
  }
  closeNote();
  syncing = true;
  if (view) maps.A.setView(view.center, view.zoom, { animate: false });
  else maps.A.fitBounds(siteBounds(0), { animate: false });
  maps.B.setView(maps.A.getCenter(), maps.A.getZoom(), { animate: false });
  syncing = false;
  refresh();
  drawNotes();
  renderNotesList();
  loadSiteExtra();
  if (S.labels) loadLabels();
}

async function loadSiteExtra() {
  const unit = S.site;
  if (!siteExtra[unit]) {
    try {
      const res = await fetch(`/api/site/${unit}`);
      if (!res.ok) return;
      siteExtra[unit] = await res.json();
    } catch (err) {
      return;
    }
  }
  if (unit === S.site) {
    PANELS.forEach(updateBg);
    showStatus();
  }
}

function setDual(on) {
  S.dual = on;
  $("panelB").hidden = !on;
  for (const k of PANELS) maps[k].invalidateSize({ animate: false });
  if (on) {
    syncing = true;
    maps.B.setView(maps.A.getCenter(), maps.A.getZoom(), { animate: false });
    syncing = false;
  }
  refresh();
  drawNotes();
  showPick();
}

/* ---------- URL hash ---------- */

function writeHash() {
  if (!maps.A || !maps.A._loaded) return;
  const c = maps.A.getCenter();
  const p = new URLSearchParams({
    s: S.site,
    a: S.run.A,
    b: S.run.B,
    ly: S.layer,
    iv: S.interval,
    y: S.year,
    bg: S.bg,
    ta: S.thr.A ?? "",
    tb: S.thr.B ?? "",
    f: S.forest ? 1 : 0,
    lb: S.labels ? 1 : 0,
    d: S.dual ? 1 : 0,
    hr: S.hires ? 1 : 0,
    o: Math.round(S.opacity * 100),
    c: `${c.lat.toFixed(5)},${c.lng.toFixed(5)},${maps.A.getZoom()}`,
  });
  history.replaceState(null, "", "#" + p.toString());
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (!p.has("s")) return null;
  S.site = p.get("s");
  S.run.A = p.get("a");
  S.run.B = p.get("b");
  if (["date", "type", "prob"].includes(p.get("ly"))) S.layer = p.get("ly");
  S.interval = p.get("iv") || "all";
  S.year = Number(p.get("y")) || null;
  if (["naip", "chm", "dchm"].includes(p.get("bg"))) S.bg = p.get("bg");
  S.thr.A = p.get("ta") ? Number(p.get("ta")) : null;
  S.thr.B = p.get("tb") ? Number(p.get("tb")) : null;
  S.forest = p.get("f") === "1";
  S.labels = p.get("lb") === "1";
  S.dual = p.get("d") !== "0";
  S.hires = p.get("hr") !== "0";
  if (p.has("o")) S.opacity = Math.min(Math.max(Number(p.get("o")) / 100, 0), 1);
  const c = (p.get("c") || "").split(",").map(Number);
  if (c.length === 3 && c.every(Number.isFinite)) return { center: [c[0], c[1]], zoom: c[2] };
  return null;
}

/* ---------- the clicked cell ---------- */

function newId() {
  return "n" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function showPick() {
  for (const k of PANELS) {
    if (pickMark[k]) pickMark[k].remove();
    pickMark[k] = null;
    if (!sel || !panels().includes(k)) continue;
    pickMark[k] = L.circleMarker([sel.lat, sel.lon], {
      pane: "notes", radius: 11, color: "#ffffff", weight: 2, fill: false, interactive: false,
    }).addTo(maps[k]);
  }
}

async function selectPoint(lat, lon, noteId) {
  await saveNow();
  if (noteId) {
    const p = notes.get(noteId).properties;
    sel = { id: noteId, lat: p.lat, lon: p.lon, isNew: false, dirty: false, props: { ...p } };
  } else {
    sel = { id: newId(), lat, lon, isNew: true, dirty: false, props: { tag: "", comment: "" } };
  }
  point = null;
  showPick();
  $("note").hidden = false;
  $("comment").value = sel.props.comment || "";
  $("note-delete").hidden = sel.isNew;
  setNoteState(sel.isNew ? "new, kept once it has a tag or text" : "saved");
  renderTags();
  renderNotesList();
  $("inspect-body").textContent = "Reading…";
  const mine = sel;
  try {
    const res = await fetch(`/api/point?unit=${S.site}&lon=${mine.lon}&lat=${mine.lat}`);
    if (sel !== mine) return;
    point = res.ok ? await res.json() : null;
  } catch (err) {
    point = null;
  }
  if (sel === mine) renderInspector();
}

function thresholdOf(name) {
  for (const k of panels()) if (S.run[k] === name) return thr(k);
  return defaultThr(CAT.runs[name]);
}

function renderInspector() {
  const box = $("inspect-body");
  if (!sel) return;
  if (!point) {
    box.textContent = "No values here.";
    return;
  }
  const names = site().runs;
  const shown = new Set(panels().map((k) => S.run[k]));
  let html = `<div class="muted">${sel.lat.toFixed(5)}, ${sel.lon.toFixed(5)}. Loss probability (%) and type per interval. Shaded cells reach the threshold.</div>`;
  html += '<table class="values"><tr><th>Interval</th>';
  for (const name of names) {
    const r = CAT.runs[name];
    const mark = panels().filter((k) => S.run[k] === name).join("+");
    html += `<th class="${shown.has(name) ? "shown" : ""}" title="${esc(r.label)} (${esc(name)})">${mark ? mark + " " : ""}${esc(runShort(r))}</th>`;
  }
  html += "</tr>";
  for (const iv of site().intervals) {
    const open = S.interval === iv ? ' style="font-weight:600"' : "";
    html += `<tr><td${open} title="${iv.replace("_", " to ")}">${intervalLabel(iv)}</td>`;
    for (const name of names) {
      const r = CAT.runs[name];
      const v = point.runs[name] || {};
      let text = "";
      let flag = false;
      if (r.family === "di") {
        flag = v.onset_year === intervalYears(iv)[1];
        text = flag ? "onset" : "";
      } else {
        const i = (v.intervals || []).indexOf(iv);
        const p = i < 0 ? null : v.prob[i];
        if (p === null || p === undefined) {
          text = '<span class="muted">–</span>';
        } else {
          flag = p >= thresholdOf(name);
          text = String(p);
          const code = v.agent ? v.agent[i] : null;
          if (code && p >= CAT.legend.prob_min) {
            text += ` <span class="dot" style="background:${agentColour(code)}"></span>${AGENT_SHORT[code]}`;
          }
        }
      }
      html += `<td class="${flag ? "flag" : ""} ${shown.has(name) ? "shown" : ""}">${text}</td>`;
    }
    html += "</tr>";
  }
  html += "</table>";
  const di = names.map((n) => point.runs[n]).find((v) => v && "onset_year" in v);
  if (di && di.max_prob !== undefined && di.max_prob !== null) {
    html += `<div class="muted">DI highest probability over intervals: ${di.max_prob}%</div>`;
  }
  box.innerHTML = html;
}

/* ---------- notes ---------- */

function setNoteState(text, bad) {
  const el = $("note-state");
  el.textContent = text ? `(${text})` : "";
  el.style.color = bad ? "#b3261e" : "";
}

function renderTags() {
  const box = $("tags");
  box.replaceChildren();
  CAT.tags.forEach((tag, i) => {
    const b = document.createElement("button");
    b.textContent = `${i + 1} ${TAG_LABELS[tag] || tag}`;
    if (sel && sel.props.tag === tag) {
      b.classList.add("on");
      b.style.background = TAG_COLOURS[tag];
    }
    b.addEventListener("click", () => setTag(tag));
    box.append(b);
  });
}

/* Mark the open note as changed. A new note takes the view it is written in. */
function touch() {
  sel.dirty = true;
  if (sel.isNew) sel.ctx = context();
}

function setTag(tag) {
  if (!sel) return;
  sel.props.tag = sel.props.tag === tag ? "" : tag;
  touch();
  renderTags();
  saveSoon(0);
}

function context() {
  return {
    unit: S.site,
    place: site().place,
    run_a: S.run.A,
    run_b: S.dual ? S.run.B : null,
    layer: S.layer,
    interval: S.interval,
    naip_year: S.year,
    background: S.bg,
    threshold_a: thr("A") / 100,
    threshold_b: S.dual ? thr("B") / 100 : null,
    forest_only: S.forest,
    zoom: maps.A.getZoom(),
  };
}

function saveSoon(delay) {
  clearTimeout(saveTimer);
  setNoteState("saving…");
  saveTimer = setTimeout(saveNow, delay);
}

async function saveNow() {
  clearTimeout(saveTimer);
  saveTimer = null;
  const s = sel;
  if (!s || !s.dirty) return;
  s.dirty = false;
  const body = { ...(s.isNew ? s.ctx : s.props), lon: s.lon, lat: s.lat, tag: s.props.tag, comment: s.props.comment };
  if (s.isNew && point) body.values = JSON.stringify(point.runs);
  try {
    const res = await fetch(`/api/notes/${s.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      keepalive: true,
    });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const out = await res.json();
    if (out.deleted) notes.delete(s.id);
    else notes.set(s.id, out);
    if (sel === s) {
      $("note-delete").hidden = !notes.has(s.id);
      setNoteState(out.deleted ? "empty, not kept" : `saved ${new Date().toLocaleTimeString()}`);
    }
  } catch (err) {
    s.dirty = true;
    if (sel === s) {
      setNoteState("NOT SAVED, trying again", true);
      saveTimer = setTimeout(saveNow, 3000);
    }
  }
  drawNotes();
  renderNotesList();
}

/* Close the open point. The state changes at once, the save runs on. */
function closeNote() {
  const saved = saveNow();
  sel = null;
  point = null;
  showPick();
  $("note").hidden = true;
  $("inspect-body").textContent = "Click the map to read every model at a cell and to leave a note there.";
  renderNotesList();
  return saved;
}

async function deleteNote() {
  if (!sel) return;
  const id = sel.id;
  clearTimeout(saveTimer);
  sel.dirty = false;
  await fetch(`/api/notes/${id}`, { method: "DELETE" });
  notes.delete(id);
  drawNotes();
  closeNote();
}

function drawNotes() {
  for (const k of PANELS) {
    noteGroup[k].clearLayers();
    if (!panels().includes(k)) continue;
    for (const f of notes.values()) {
      const p = f.properties;
      if (p.unit !== S.site) continue;
      const m = L.circleMarker([p.lat, p.lon], {
        pane: "notes",
        radius: 6,
        color: "#ffffff",
        weight: 1.5,
        fillColor: TAG_COLOURS[p.tag] || "#888888",
        fillOpacity: 1,
      });
      m.bindTooltip(`${TAG_LABELS[p.tag] || "Note"}: ${esc((p.comment || "").slice(0, 120))}`);
      m.on("click", (e) => {
        L.DomEvent.stopPropagation(e);
        selectPoint(p.lat, p.lon, p.id);
      });
      m.addTo(noteGroup[k]);
    }
  }
}

function renderNotesList() {
  const all = $("notes-all").checked;
  const props = [...notes.values()].map((f) => f.properties);
  const scope = props.filter((p) => all || p.unit === S.site);
  const here = props.filter((p) => p.unit === S.site).length;
  $("notes-count").textContent = `${here} here, ${props.length} in all`;
  const counts = $("tag-counts");
  counts.replaceChildren();
  for (const tag of CAT.tags) {
    const n = scope.filter((p) => p.tag === tag).length;
    if (!n) continue;
    const b = document.createElement("button");
    b.textContent = `${TAG_LABELS[tag]} ${n}`;
    if (tagFilter === tag) {
      b.classList.add("on");
      b.style.background = TAG_COLOURS[tag];
    }
    b.addEventListener("click", () => {
      tagFilter = tagFilter === tag ? "" : tag;
      renderNotesList();
    });
    counts.append(b);
  }
  const list = $("notes-list");
  list.replaceChildren();
  const rows = scope
    .filter((p) => !tagFilter || p.tag === tagFilter)
    .sort((a, b) => String(b.updated).localeCompare(String(a.updated)));
  for (const p of rows) {
    const li = document.createElement("li");
    if (sel && sel.id === p.id) li.classList.add("open");
    const where = [all ? p.place : "", p.interval && p.interval !== "all" ? intervalLabel(p.interval) : "", p.layer]
      .filter(Boolean)
      .join(" · ");
    li.innerHTML =
      `<span class="dot" style="background:${TAG_COLOURS[p.tag] || "#888"}"></span>` +
      `<b>${esc(TAG_LABELS[p.tag] || "Note")}</b> ${esc(p.comment)}<small>${esc(where)}</small>`;
    li.addEventListener("click", () => gotoNote(p.id));
    list.append(li);
  }
}

/* Return to the view a note was written in, and open it. */
async function gotoNote(id) {
  const p = notes.get(id).properties;
  if (!SITES[p.unit]) return;
  await saveNow();
  S.site = p.unit;
  const runs = SITES[p.unit].runs;
  if (runs.includes(p.run_a)) S.run.A = p.run_a;
  if (runs.includes(p.run_b)) S.run.B = p.run_b;
  if (p.layer) S.layer = p.layer;
  S.interval = p.interval || "all";
  if (p.naip_year) S.year = p.naip_year;
  if (p.background) S.bg = p.background;
  S.thr.A = typeof p.threshold_a === "number" && p.threshold_a > 0 ? Math.round(p.threshold_a * 100) : null;
  S.thr.B = typeof p.threshold_b === "number" && p.threshold_b > 0 ? Math.round(p.threshold_b * 100) : null;
  S.forest = Boolean(p.forest_only);
  applySite({ center: [p.lat, p.lon], zoom: p.zoom || 15 });
  selectPoint(p.lat, p.lon, id);
}

async function loadNotes() {
  const res = await fetch("/api/notes");
  const data = await res.json();
  notes.clear();
  for (const f of data.features) notes.set(f.properties.id, f);
}

/* ---------- wiring ---------- */

function wire() {
  $("site").addEventListener("change", (e) => {
    S.site = e.target.value;
    e.target.blur();
    applySite();
  });
  for (const b of $("layer").children) {
    b.addEventListener("click", () => {
      S.layer = b.dataset.layer;
      b.blur();
      refresh();
    });
  }
  const check = (id, key, after) =>
    $(id).addEventListener("change", (e) => {
      S[key] = e.target.checked;
      e.target.blur();
      refresh();
      if (after) after();
    });
  check("forest", "forest");
  check("hires", "hires");
  check("labels", "labels", () => S.labels && loadLabels());
  $("opacity").addEventListener("input", (e) => {
    S.opacity = Number(e.target.value) / 100;
    S.overlay = true;
    PANELS.forEach(updateOverlay);
  });
  $("opacity").addEventListener("change", (e) => {
    e.target.blur();
    writeHash();
  });
  $("bg").addEventListener("change", (e) => {
    S.bg = e.target.value;
    e.target.blur();
    refresh();
  });
  $("dual").addEventListener("click", (e) => {
    e.target.blur();
    setDual(!S.dual);
  });
  $("help").addEventListener("click", () => {
    $("help-box").hidden = !$("help-box").hidden;
  });
  for (const k of PANELS) {
    const el = $("panel" + k);
    el.querySelector(".run").addEventListener("change", (e) => {
      S.run[k] = e.target.value;
      S.thr[k] = null;
      e.target.blur();
      refresh();
    });
    const slider = el.querySelector(".thr input");
    slider.addEventListener("input", () => {
      S.thr[k] = Number(slider.value);
      el.querySelector(".thr output").textContent = (S.thr[k] / 100).toFixed(2);
    });
    slider.addEventListener("change", () => {
      slider.blur();
      refresh();
    });
    el.querySelector(".reset").addEventListener("click", (e) => {
      S.thr[k] = null;
      e.target.blur();
      refresh();
    });
  }
  $("comment").addEventListener("input", (e) => {
    if (!sel) return;
    sel.props.comment = e.target.value;
    touch();
    saveSoon(SAVE_DELAY_MS);
  });
  $("note-close").addEventListener("click", closeNote);
  $("note-delete").addEventListener("click", deleteNote);
  $("notes-all").addEventListener("change", renderNotesList);
  window.addEventListener("beforeunload", () => {
    saveNow();
  });
  window.addEventListener("keydown", onKey);
  window.addEventListener("keyup", (e) => {
    if (e.code === "Space" && peek) {
      peek = false;
      PANELS.forEach(updateOverlay);
    }
  });
}

function onKey(e) {
  if (e.key === "Escape") {
    $("help-box").hidden = true;
    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
    closeNote();
    return;
  }
  const typing = e.target.matches && e.target.matches("textarea, input[type=text], select");
  if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
  const s = site();
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (e.code === "Space") {
    e.preventDefault();
    if (!peek) {
      peek = true;
      PANELS.forEach(updateOverlay);
    }
    return;
  }
  if (key >= "1" && key <= "6" && sel) {
    setTag(CAT.tags[Number(key) - 1]);
    return;
  }
  switch (key) {
    case ",":
    case ".":
      S.site = stepList(CAT.sites.map((x) => x.unit), S.site, key === "," ? -1 : 1);
      applySite();
      break;
    case "ArrowLeft":
    case "ArrowRight":
      S.year = stepList(s.years.map((y) => y.year), S.year, key === "ArrowLeft" ? -1 : 1);
      refresh();
      break;
    case "[":
    case "]":
      setInterval_(stepList(["all", ...s.intervals], S.interval, key === "[" ? -1 : 1));
      break;
    case "b":
    case "a":
      if (S.interval !== "all") {
        S.year = intervalYears(S.interval)[key === "b" ? 0 : 1];
        refresh();
      }
      break;
    case "d":
      S.layer = "date";
      refresh();
      break;
    case "t":
      S.layer = "type";
      refresh();
      break;
    case "p":
      S.layer = "prob";
      refresh();
      break;
    case "o":
      S.overlay = !S.overlay;
      PANELS.forEach(updateOverlay);
      break;
    case "l":
      S.labels = !S.labels;
      refresh();
      if (S.labels) loadLabels();
      break;
    case "s":
      setDual(!S.dual);
      break;
    case "n":
      if (lastLatLng) {
        e.preventDefault();
        selectPoint(lastLatLng.lat, lastLatLng.lng).then(() => $("comment").focus());
      }
      break;
    case "?":
      $("help-box").hidden = !$("help-box").hidden;
      break;
    default:
      return;
  }
  e.preventDefault();
}

async function init() {
  const res = await fetch("/api/catalog");
  CAT = await res.json();
  if (!CAT.sites.length) {
    document.body.innerHTML = "<p style='padding:20px'>No pair-model maps were found in the export folder.</p>";
    return;
  }
  for (const s of CAT.sites) {
    SITES[s.unit] = s;
    $("site").append(new Option(`${s.place} (${s.unit})`, s.unit));
  }
  $("notes-path").textContent = `Saved in ${CAT.notes_path}`;
  if (!CAT.labels) $("labels").parentElement.hidden = true;
  const view = readHash();
  if (!SITES[S.site]) S.site = CAT.sites[0].unit;
  $("panelB").hidden = !S.dual;
  PANELS.forEach(makeMap);
  wire();
  await loadNotes();
  applySite(view);
  // ?pick=lat,lon opens a cell on load (a link to a place, and for tests).
  const pick = (new URLSearchParams(location.search).get("pick") || "").split(",").map(Number);
  if (pick.length === 2 && pick.every(Number.isFinite)) selectPoint(pick[0], pick[1]);
}

init();
