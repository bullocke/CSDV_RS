/* Blind audit page. Plain JavaScript, no dependencies.
 *
 * Served by `csdv audit app`, answers go to the server (POST answer). Opened
 * from disk, answers stay in the browser and are saved with "Download
 * answers". The item list comes from items.js.
 */
(function () {
  "use strict";

  const ITEMS = window.AUDIT_ITEMS || [];
  const META = window.AUDIT_META || { name: "audit" };
  const SERVED = location.protocol === "http:" || location.protocol === "https:";
  const STORE_KEY = "csdv-audit:" + META.name;
  const ONLY_BLOCK = Number((/[?&]block=(\d+)/.exec(location.search) || [])[1] || 0);
  const TYPES = [
    ["h", "harvest", "harvest"],
    ["w", "wind", "wind"],
    ["f", "fire", "fire"],
    ["m", "biotic_mortality", "mortality (insects, disease)"],
    ["c", "conversion", "conversion"],
    ["o", "other", "other"],
  ];
  const OUTLINES = [
    ["g", "good", "label holds"],
    ["l", "large", "too large"],
    ["s", "small", "too small"],
    ["x", "wrong", "wrong place"],
  ];
  const INTRO = {
    1:
      "<h1>Block 1: outlined areas</h1>" +
      "<p>Each item shows the same ground on every date. Look inside the yellow outline.</p>" +
      "<p>Did live forest canopy disappear between two dates, over about 0.1 ha or more, " +
      "even if that is only part of the outline? Press the number of the <b>first image in " +
      "which the loss is visible</b>, or <b>0</b> if there is none.</p>" +
      "<p>Not a loss: differences of colour, shadow, season or haze, leafless trees, " +
      "regrowth, ground that was already open in image 1, and dead trees that fall or " +
      "are removed later.</p>" +
      "<p>Arrow keys enlarge an image and flick between dates. Esc returns to all dates.</p>",
    2:
      "<h1>Block 2: whole windows</h1>" +
      "<p>The label is drawn on the image. Red is labelled loss, hatched ground is left " +
      "out of training, darkened ground has no label, and clear ground is labelled " +
      "background (no loss).</p>" +
      "<p>Look for canopy loss <b>on clear ground</b>, in any interval. Press <b>space</b> " +
      "if there is none. Otherwise click on each loss the label missed, in the first " +
      "image that shows it, then press <b>Enter</b>.</p>" +
      "<p>Arrow keys flick between dates, the wheel zooms, T hides the label.</p>",
    3:
      "<h1>Block 3: more outlined areas</h1>" +
      "<p>The same question as block 1, on other ground. Look inside the yellow outline, " +
      "which may follow a grid of small squares.</p>" +
      "<p>Did live forest canopy disappear between two dates, over about 0.1 ha or more? " +
      "Press the number of the <b>first image in which the loss is visible</b>, or <b>0</b> " +
      "if there is none.</p>" +
      "<p>Not a loss: differences of colour, shadow, season or haze, leafless trees, " +
      "regrowth, ground that was already open in image 1, and dead trees that fall or " +
      "are removed later.</p>",
  };

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, html) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (html !== undefined) node.innerHTML = html;
    return node;
  };

  const state = {
    records: [],
    latest: new Map(),
    interpreter: "",
    item: null,
    step: 1,
    draft: null,
    index: 1,
    viewing: false,
    showLabel: true,
    started: 0,
    hiddenAt: 0,
    gate: null,
    introduced: new Set(),
    reserveOk: false,
    saving: false,
  };
  const view = { scale: 1, x: 0, y: 0, side: 0, drag: null };
  const keep = new Map();

  /* ---------- storage ---------- */

  function rebuildLatest() {
    state.latest = new Map();
    for (const r of state.records) {
      if (r.void) state.latest.delete(r.item);
      else state.latest.set(r.item, r);
    }
  }

  function persistLocal() {
    if (SERVED) return;
    try {
      localStorage.setItem(
        STORE_KEY,
        JSON.stringify({ interpreter: state.interpreter, records: state.records })
      );
    } catch (err) {
      setStatus("This browser cannot keep answers. Download them often.", true);
    }
  }

  async function loadState() {
    if (SERVED) {
      const response = await fetch("state");
      const body = await response.json();
      state.interpreter = body.interpreter;
      state.records = body.answers;
    } else {
      let stored = {};
      try {
        stored = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
      } catch (err) {
        stored = {};
      }
      state.records = stored.records || [];
      // index.html?who=NAME names the interpreter without the dialog.
      const who = /[?&]who=([^&]+)/.exec(location.search);
      state.interpreter = stored.interpreter || (who ? decodeURIComponent(who[1]) : "");
      if (!state.interpreter) {
        state.interpreter = (window.prompt("Your name or initials") || "anonymous").trim();
      }
      persistLocal();
    }
    rebuildLatest();
  }

  async function store(record) {
    if (SERVED) {
      const response = await fetch("answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(record),
      });
      if (!response.ok) throw new Error("server said " + response.status);
      const body = await response.json();
      record.saved = body.saved;
    } else {
      record.saved = new Date().toISOString();
    }
    record.interpreter = state.interpreter;
    state.records.push(record);
    rebuildLatest();
    persistLocal();
  }

  function download() {
    const lines = state.records.map((r) => JSON.stringify(r)).join("\n") + "\n";
    const link = el("a");
    link.href = URL.createObjectURL(new Blob([lines], { type: "application/x-ndjson" }));
    link.download = "answers_" + state.interpreter.replace(/[^\w-]+/g, "_") + ".jsonl";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  /* ---------- header ---------- */

  function setStatus(text, bad) {
    $("status").textContent = text;
    $("status").className = bad ? "bad" : "";
  }

  function median(values) {
    if (!values.length) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function minutes(seconds) {
    const m = Math.round(seconds / 60);
    return m < 60 ? m + " min" : Math.floor(m / 60) + " h " + (m % 60) + " min";
  }

  function header() {
    const item = state.item;
    const block = item ? item.block : 0;
    const mine = ITEMS.filter((i) => i.block === block);
    const target = mine.filter((i) => !i.reserve);
    const doneTarget = target.filter((i) => state.latest.has(i.id)).length;
    const done = mine.filter((i) => state.latest.has(i.id)).length;
    let text = "Block " + block + " · " + doneTarget + " of " + target.length;
    if (done > doneTarget) text += " + " + (done - doneTarget) + " reserve";
    $("progress").textContent = item ? text : "";
    const times = mine
      .map((i) => state.latest.get(i.id))
      .filter((r) => r && r.seconds < 600)
      .map((r) => r.seconds);
    const pace = median(times);
    const left = target.length - doneTarget;
    $("pace").textContent = pace
      ? Math.round(pace) + " s per item · about " + minutes(pace * left) + " left"
      : "";
    $("where").textContent = item
      ? [item.metres + " m across", item.place].filter(Boolean).join(" · ")
      : "";
    $("who").textContent = state.interpreter;
  }

  /* ---------- hints ---------- */

  function hintButton(key, label, action, quiet) {
    const button = el("button", quiet ? "quiet" : "");
    button.innerHTML = '<span class="key">' + key + "</span>" + label;
    button.addEventListener("click", () => {
      action();
      button.blur();
    });
    return button;
  }

  function hints(ask, buttons) {
    const bar = $("hints");
    bar.innerHTML = "";
    bar.appendChild(el("span", "ask", ask));
    for (const b of buttons) bar.appendChild(b);
    bar.appendChild(el("span", "sep"));
    bar.appendChild(hintButton("←→", "dates", () => flick(1), true));
    if (state.item && (state.item.kind === "window" || state.step === 3)) {
      bar.appendChild(hintButton("T", "label on/off", toggleLabel, true));
    }
    bar.appendChild(hintButton("/", "note", note, true));
    bar.appendChild(hintButton("Z", "previous item", undo, true));
  }

  function renderHints() {
    const item = state.item;
    if (!item) return;
    const n = item.imgs.length;
    if (item.kind === "window") {
      const marks = state.draft.marks.length;
      hints(
        marks ? marks + " missed loss marked" : "Any canopy loss on clear ground?",
        marks
          ? [
              hintButton("Enter", "save", () => finishWindow("loss")),
              hintButton("⌫", "remove last mark", removeMark),
            ]
          : [
              hintButton("space", "none", () => finishWindow("none")),
              hintButton("click", "mark a missed loss", () => {}),
              hintButton("U", "cannot tell", () => finishWindow("unsure")),
            ]
      );
      return;
    }
    if (state.step === 1) {
      const buttons = [hintButton("0", "no loss", () => answerLoss(0))];
      for (let i = 2; i <= n; i++) {
        buttons.push(hintButton(String(i), "", () => answerLoss(i)));
      }
      buttons.push(hintButton("U", "cannot tell", () => answerLoss("u")));
      hints("Canopy loss inside the outline? First image that shows it:", buttons);
    } else if (state.step === 2) {
      const buttons = TYPES.map(([key, code, label]) =>
        hintButton(key.toUpperCase(), label, () => answerType(code))
      );
      buttons.push(hintButton("⌫", "back", back, true));
      hints("Loss first seen in image " + state.draft.loss_at + ". Cause?", buttons);
    } else {
      const label = item.labels[0];
      const buttons = OUTLINES.map(([key, code, text]) =>
        hintButton(key.toUpperCase(), text, () => answerOutline(code))
      );
      buttons.push(hintButton("⌫", "back", back, true));
      hints(
        "The label says loss (red) between image " +
          (label.image - 1) +
          " and " +
          label.image +
          ". Hatched is left out, dark has no label. Does the red extent hold?",
        buttons
      );
    }
  }

  /* ---------- viewer ---------- */

  function buildViewer(item) {
    const canvas = $("canvas");
    canvas.innerHTML = "";
    item.imgs.forEach((src, i) => {
      const img = el("img", "layer");
      img.src = src;
      img.dataset.i = i + 1;
      img.draggable = false;
      canvas.appendChild(img);
    });
    for (const label of item.labels) {
      const img = el("img", "overlay");
      img.src = label.png;
      img.dataset.i = label.image;
      img.draggable = false;
      canvas.appendChild(img);
    }
    const marks = el("div");
    marks.id = "marks";
    canvas.appendChild(marks);
    const strip = $("strip");
    strip.innerHTML = "";
    item.imgs.forEach((src, i) => {
      const thumb = el("div", "thumb");
      const img = el("img");
      img.src = src;
      img.draggable = false;
      thumb.appendChild(img);
      thumb.appendChild(el("span", "", i + 1 + " · " + item.dates[i]));
      thumb.addEventListener("click", () => setIndex(i + 1));
      strip.appendChild(thumb);
    });
    let tag = $("tag");
    if (!tag) {
      tag = el("div");
      tag.id = "tag";
      $("stage").appendChild(tag);
    }
    view.scale = 1;
    view.x = 0;
    view.y = 0;
  }

  function labelsAllowed() {
    return state.item.kind === "window" || state.step === 3;
  }

  function setIndex(n) {
    const item = state.item;
    const count = item.imgs.length;
    state.index = Math.min(Math.max(n, 1), count);
    const canvas = $("canvas");
    for (const img of canvas.querySelectorAll("img.layer")) {
      img.classList.toggle("on", Number(img.dataset.i) === state.index);
    }
    const show = state.showLabel && labelsAllowed();
    for (const img of canvas.querySelectorAll("img.overlay")) {
      img.classList.toggle("on", show && Number(img.dataset.i) === state.index);
    }
    [...$("strip").children].forEach((thumb, i) =>
      thumb.classList.toggle("on", i + 1 === state.index)
    );
    $("tag").textContent = state.index + " · " + item.dates[state.index - 1];
    renderMarks();
  }

  function layoutViewer() {
    const stage = $("stage");
    view.side = Math.max(Math.min(stage.clientWidth, stage.clientHeight) - 4, 100);
    const canvas = $("canvas");
    canvas.style.width = view.side + "px";
    canvas.style.height = view.side + "px";
    applyView();
  }

  function applyView() {
    const limit = view.side * (view.scale - 1);
    view.x = Math.min(0, Math.max(-limit, view.x));
    view.y = Math.min(0, Math.max(-limit, view.y));
    $("canvas").style.transform =
      "translate(" + view.x + "px," + view.y + "px) scale(" + view.scale + ")";
  }

  function openViewer(n) {
    state.viewing = true;
    $("grid").hidden = true;
    $("viewer").hidden = false;
    layoutViewer();
    setIndex(n);
  }

  function closeViewer() {
    if (!state.item || state.item.kind !== "cand" || state.step === 3) return;
    state.viewing = false;
    $("viewer").hidden = true;
    $("grid").hidden = false;
    layoutGrid();
  }

  function flick(step) {
    if (!state.item) return;
    if (!state.viewing) {
      openViewer(state.index);
      return;
    }
    setIndex(state.index + step);
  }

  function toggleLabel() {
    state.showLabel = !state.showLabel;
    if (state.viewing) setIndex(state.index);
  }

  function onWheel(event) {
    if (!state.viewing) return;
    event.preventDefault();
    const rect = $("canvas").getBoundingClientRect();
    const fx = (event.clientX - rect.left) / rect.width;
    const fy = (event.clientY - rect.top) / rect.height;
    const next = Math.min(8, Math.max(1, view.scale * (event.deltaY < 0 ? 1.25 : 0.8)));
    view.x -= fx * view.side * (next - view.scale);
    view.y -= fy * view.side * (next - view.scale);
    view.scale = next;
    applyView();
  }

  function onPointerDown(event) {
    if (event.button !== 0) return;
    view.drag = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y, moved: false };
  }

  function onPointerMove(event) {
    const drag = view.drag;
    if (!drag) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 5) drag.moved = true;
    if (drag.moved && view.scale > 1) {
      view.x = drag.vx + dx;
      view.y = drag.vy + dy;
      applyView();
    }
  }

  function onPointerUp(event) {
    const drag = view.drag;
    view.drag = null;
    if (!drag || drag.moved || !state.item || state.item.kind !== "window") return;
    if (state.gate) return;
    const rect = $("canvas").getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    state.draft.marks.push({
      x: Math.round(x * 10000) / 10000,
      y: Math.round(y * 10000) / 10000,
      image: state.index,
    });
    renderMarks();
    renderHints();
  }

  function renderMarks() {
    const layer = $("marks");
    if (!layer || !state.draft) return;
    layer.innerHTML = "";
    for (const mark of state.draft.marks || []) {
      const dot = el("div", "mark" + (mark.image === state.index ? "" : " other"));
      dot.style.left = mark.x * 100 + "%";
      dot.style.top = mark.y * 100 + "%";
      layer.appendChild(dot);
    }
  }

  function removeMark() {
    state.draft.marks.pop();
    renderMarks();
    renderHints();
  }

  /* ---------- grid of a candidate item ---------- */

  function buildGrid(item) {
    const grid = $("grid");
    grid.innerHTML = "";
    item.imgs.forEach((src, i) => {
      const cell = el("figure", "cell");
      const img = el("img");
      img.src = src;
      img.draggable = false;
      cell.appendChild(img);
      cell.appendChild(el("span", "num", String(i + 1)));
      cell.addEventListener("click", () => openViewer(i + 1));
      grid.appendChild(cell);
    });
  }

  function layoutGrid() {
    const item = state.item;
    if (!item || item.kind !== "cand") return;
    const n = item.imgs.length;
    const main = $("main");
    const gap = 8;
    let best = { side: 0, cols: n };
    for (let cols = 1; cols <= n; cols++) {
      const rows = Math.ceil(n / cols);
      const side = Math.floor(
        Math.min(
          (main.clientWidth - gap * (cols + 1)) / cols,
          (main.clientHeight - gap * (rows + 1)) / rows
        )
      );
      if (side > best.side) best = { side: side, cols: cols };
    }
    const grid = $("grid");
    grid.style.gridTemplateColumns = "repeat(" + best.cols + ", " + best.side + "px)";
    grid.style.gridAutoRows = best.side + "px";
  }

  function markCells() {
    const cells = [...$("grid").children];
    cells.forEach((cell, i) => {
      const n = i + 1;
      cell.classList.toggle("first", state.draft.loss_at === n);
      cell.classList.toggle("more", state.draft.more_at.includes(n));
    });
  }

  /* ---------- flow ---------- */

  function preload(items) {
    keep.clear();
    for (const item of items) {
      for (const src of item.imgs.concat(item.labels.map((l) => l.png))) {
        const img = new Image();
        img.src = src;
        keep.set(src, img);
      }
    }
  }

  // Items still to judge, in serving order: the target items of every
  // block first, the reserve after them. index.html?block=2 keeps to one block.
  function pending() {
    const open = ITEMS.filter(
      (i) => !state.latest.has(i.id) && (!ONLY_BLOCK || i.block === ONLY_BLOCK)
    );
    const target = open.filter((i) => !i.reserve);
    return target.length ? target : open;
  }

  function message(html, onGo) {
    state.gate = onGo || (() => {});
    $("grid").hidden = true;
    $("viewer").hidden = true;
    const box = $("message");
    box.innerHTML = "<div>" + html + "</div>";
    box.hidden = false;
    $("hints").innerHTML = "";
  }

  function closeGate() {
    const go = state.gate;
    state.gate = null;
    $("message").hidden = true;
    go();
  }

  function next() {
    const queue = pending();
    const item = queue[0];
    if (!item) {
      state.item = null;
      header();
      message(
        "<h1>All items are judged</h1><p>Thank you. " +
          (SERVED
            ? "Every answer is stored."
            : "Press <b>Download answers</b> and send the file back.") +
          "</p>"
      );
      state.gate = null;
      return;
    }
    if (item.reserve && !state.reserveOk) {
      state.reserveOk = true;
      message(
        "<h1>Every target item is judged</h1><p>The items that follow are a reserve " +
          "from the same draw. Each one judged narrows the intervals. Stop here " +
          "whenever you like.</p>" +
          '<p class="go">Press Enter to continue with the reserve.</p>',
        () => next()
      );
      return;
    }
    if (!state.introduced.has(item.block)) {
      state.introduced.add(item.block);
      message(INTRO[item.block] + '<p class="go">Press Enter to start.</p>', () => show(item));
      return;
    }
    show(item);
  }

  function show(item) {
    state.item = item;
    state.step = 1;
    state.index = 1;
    state.showLabel = true;
    state.draft =
      item.kind === "window"
        ? { marks: [], note: "" }
        : { loss_at: null, more_at: [], type: null, outline: null, note: "" };
    $("message").hidden = true;
    buildViewer(item);
    if (item.kind === "cand") {
      buildGrid(item);
      state.viewing = false;
      $("viewer").hidden = true;
      $("grid").hidden = false;
      layoutGrid();
      markCells();
    } else {
      openViewer(Math.min(2, item.imgs.length));
    }
    state.started = performance.now();
    setStatus("");
    header();
    renderHints();
    preload(pending().filter((i) => i.id !== item.id).slice(0, 2));
  }

  function seconds() {
    return Math.round((performance.now() - state.started) / 100) / 10;
  }

  async function save(record) {
    if (state.saving) return;
    state.saving = true;
    record.item = state.item.id;
    record.kind = state.item.kind;
    record.seconds = seconds();
    record.note = state.draft.note || "";
    try {
      await store(record);
    } catch (err) {
      state.saving = false;
      setStatus("Not saved (" + err.message + "). Press the key again.", true);
      return;
    }
    state.saving = false;
    next();
  }

  function answerLoss(value) {
    if (state.item.kind !== "cand" || state.step !== 1) return;
    if (value === 0) {
      save({ answer: "none", loss_at: null, more_at: [], type: null, outline: null });
    } else if (value === "u") {
      save({ answer: "unsure", loss_at: null, more_at: [], type: null, outline: null });
    } else if (value >= 2 && value <= state.item.imgs.length) {
      state.draft.loss_at = value;
      state.step = 2;
      markCells();
      renderHints();
    }
  }

  function addLoss(value) {
    const draft = state.draft;
    if (state.step !== 2 || value <= draft.loss_at || value > state.item.imgs.length) return;
    const at = draft.more_at.indexOf(value);
    if (at >= 0) draft.more_at.splice(at, 1);
    else draft.more_at.push(value);
    draft.more_at.sort((a, b) => a - b);
    markCells();
  }

  function answerType(code) {
    if (state.step !== 2) return;
    state.draft.type = code;
    if (state.item.labels.length) {
      state.step = 3;
      state.showLabel = true;
      openViewer(state.item.labels[0].image);
      renderHints();
    } else {
      finishCand();
    }
  }

  function answerOutline(code) {
    if (state.step !== 3) return;
    state.draft.outline = code;
    finishCand();
  }

  function finishCand() {
    const d = state.draft;
    save({
      answer: "loss",
      loss_at: d.loss_at,
      more_at: d.more_at,
      type: d.type,
      outline: d.outline,
    });
  }

  function finishWindow(answer) {
    if (state.item.kind !== "window") return;
    const marks = state.draft.marks;
    if (answer === "loss" && !marks.length) return;
    if (answer === "none" && marks.length) return;
    save({ answer: answer, marks: answer === "loss" ? marks : [] });
  }

  function back() {
    if (!state.item || state.item.kind !== "cand" || state.step === 1) return;
    if (state.step === 3) {
      state.step = 2;
      state.draft.type = null;
      closeViewerForced();
    } else {
      state.step = 1;
      state.draft.loss_at = null;
      state.draft.more_at = [];
    }
    markCells();
    renderHints();
  }

  function closeViewerForced() {
    state.viewing = false;
    $("viewer").hidden = true;
    $("grid").hidden = false;
    layoutGrid();
  }

  async function undo() {
    if (state.saving || state.gate) return;
    let last = null;
    for (let i = state.records.length - 1; i >= 0; i--) {
      const r = state.records[i];
      if (!r.void && state.latest.get(r.item) === r) {
        last = r;
        break;
      }
    }
    if (!last) return;
    const item = ITEMS.find((i) => i.id === last.item);
    if (!item) return;
    state.saving = true;
    try {
      await store({ item: last.item, void: true });
    } catch (err) {
      state.saving = false;
      setStatus("Undo failed (" + err.message + ").", true);
      return;
    }
    state.saving = false;
    show(item);
    setStatus("Judging the previous item again");
  }

  function note() {
    if (!state.item || !state.draft) return;
    const text = window.prompt("Note for this item", state.draft.note || "");
    if (text !== null) state.draft.note = text.trim();
  }

  /* ---------- keys ---------- */

  function digit(event) {
    const match = /^(?:Digit|Numpad)(\d)$/.exec(event.code || "");
    if (match) return Number(match[1]);
    return /^\d$/.test(event.key) ? Number(event.key) : null;
  }

  function onKey(event) {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (state.gate) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        closeGate();
      }
      return;
    }
    const item = state.item;
    if (!item) return;
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    if (key === "ArrowRight" || key === "ArrowLeft") {
      event.preventDefault();
      flick(key === "ArrowRight" ? 1 : -1);
      return;
    }
    if (key === "Escape" || key === "ArrowUp" || key === "ArrowDown") {
      event.preventDefault();
      closeViewer();
      return;
    }
    if (key === "z") return void undo();
    if (key === "/") {
      event.preventDefault();
      return void note();
    }
    if (key === "t" && labelsAllowed()) return void toggleLabel();
    if (item.kind === "window") {
      if (key === " " || digit(event) === 0) {
        event.preventDefault();
        finishWindow("none");
      } else if (key === "Enter") finishWindow("loss");
      else if (key === "u") finishWindow("unsure");
      else if (key === "Backspace") removeMark();
      return;
    }
    if (key === "Backspace") {
      event.preventDefault();
      return void back();
    }
    const number = digit(event);
    if (state.step === 1) {
      if (key === " ") {
        event.preventDefault();
        answerLoss(0);
      } else if (number !== null && !event.shiftKey) answerLoss(number);
      else if (key === "u") answerLoss("u");
    } else if (state.step === 2) {
      if (number !== null && event.shiftKey) return void addLoss(number);
      const type = TYPES.find((t) => t[0] === key);
      if (type) answerType(type[1]);
    } else {
      const outline = OUTLINES.find((o) => o[0] === key);
      if (outline) answerOutline(outline[1]);
    }
  }

  /* ---------- start ---------- */

  document.addEventListener("keydown", onKey);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) state.hiddenAt = performance.now();
    else if (state.hiddenAt) {
      state.started += performance.now() - state.hiddenAt;
      state.hiddenAt = 0;
    }
  });
  window.addEventListener("resize", () => {
    if (!state.item || state.gate) return;
    if (state.viewing) layoutViewer();
    else layoutGrid();
  });
  $("stage").addEventListener("wheel", onWheel, { passive: false });
  $("stage").addEventListener("pointerdown", onPointerDown);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  $("download").addEventListener("click", download);

  // Test hook: ?keys=2,h,g replays a key sequence once the first item shows.
  // click@0.4@0.6 clicks the large image at that fraction of its width and height.
  function replay() {
    const match = /[?&]keys=([^&]+)/.exec(location.search);
    if (!match) return;
    const keys = decodeURIComponent(match[1]).split(",");
    const press = () => {
      if (!keys.length) {
        document.title = "replayed";
        return;
      }
      if (state.saving) return void setTimeout(press, 30);
      const key = keys.shift();
      const click = /^click@([\d.]+)@([\d.]+)$/.exec(key);
      if (click) {
        const rect = $("canvas").getBoundingClientRect();
        const at = {
          button: 0,
          clientX: rect.left + Number(click[1]) * rect.width,
          clientY: rect.top + Number(click[2]) * rect.height,
        };
        $("stage").dispatchEvent(new MouseEvent("pointerdown", at));
        window.dispatchEvent(new MouseEvent("pointerup", at));
        return void setTimeout(press, 60);
      }
      const code = /^\d$/.test(key) ? "Digit" + key : "";
      document.dispatchEvent(new KeyboardEvent("keydown", { key: key, code: code }));
      setTimeout(press, 60);
    };
    press();
  }

  loadState()
    .then(() => {
      header();
      next();
      replay();
    })
    .catch((err) => {
      message("<h1>The page could not start</h1><p>" + err.message + "</p>");
      state.gate = null;
    });
})();
