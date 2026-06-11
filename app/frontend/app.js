import { WebSocketRecognizer } from "./websocket_recognizer.js?v=1";

/* ─────────────────────────────────────────────────────────────────────────

   Router + cámara compartida + 4 vistas:
     · Reconocer    → HUD en streaming (WebSocket /ws/predict)
     · Grabar       → captura muestras al dataset (POST /collect/*)
     · Entrenar     → re-entrena el modelo (POST /train, GET /train/status)
     · Inspeccionar → estado del dataset (GET /api/dataset)

   No contiene lógica de reconocimiento: solo consume la API del backend.
   ───────────────────────────────────────────────────────────────────────── */

// ── Constantes de captura ───────────────────────────────────────────────────
const TARGET_FPS = 30;
const TARGET_INTERVAL_MS = 1000 / TARGET_FPS;
// En modo ligero se manda un JPEG más chico y de menor calidad. Eso baja el
// costo del worker y del backend, aunque puede perder detalle en casos difíciles.
const SEND_FULL = { width: 800, quality: 0.78 };
const SEND_LITE = { width: 640, quality: 0.7 };
const HISTORY_MAX = 8;

// ── DOM compartido ──────────────────────────────────────────────────────────
const stage = document.getElementById("stage");
const video = document.getElementById("video");
const landmarksCanvas = document.getElementById("landmarks");
const particlesCanvas = document.getElementById("particles");
const radarCanvas = document.getElementById("radar");
const backBtn = document.getElementById("back-btn");
const statusTextEl = document.getElementById("status-text");
const fpsEl = document.getElementById("fps");
const perfBtn = document.getElementById("perf-btn");
const voiceBtn = document.getElementById("voice-btn");

// Reconocer
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const translationTextEl = document.getElementById("translation-text");
const barsEl = document.getElementById("confidence-bars");
const motionLabelEl = document.getElementById("motion-label");
const historyEl = document.getElementById("history");
const predictionBox = document.getElementById("prediction-box");

// Grabar
const labelInput = document.getElementById("collect-label");
const labelList = document.getElementById("label-list");
const recordBtn = document.getElementById("collect-record");
const stopBtn = document.getElementById("collect-stop");
const discardBtn = document.getElementById("collect-discard");
const framesEl = document.getElementById("collect-frames");
const detectedEl = document.getElementById("collect-detected");
const collectCountsEl = document.getElementById("collect-counts");
const countdownEl = document.getElementById("countdown");
const toastEl = document.getElementById("collect-toast");

// Entrenar / Inspeccionar
const trainSummaryEl = document.getElementById("train-dataset-summary");
const trainStartBtn = document.getElementById("train-start");
const trainStatusEl = document.getElementById("train-status");
const trainMetricsEl = document.getElementById("train-metrics");
const inspectSummaryEl = document.getElementById("inspect-summary");
const inspectBodyEl = document.getElementById("inspect-body");
const inspectRefreshBtn = document.getElementById("inspect-refresh");

// ── Estado ───────────────────────────────────────────────────────────────────
let currentView = "menu";
const hud = {
  state: "WAITING",
  velocity: 0,
  motionThreshold: 0.03,
  accepted: false,
};
let acceptedActive = false;
let lastRejectedKey = "";

// ── Captura de frames (compartida) ───────────────────────────────────────────
const grabCanvas = document.createElement("canvas");
const grabCtx = grabCanvas.getContext("2d");

function captureBlob() {
  return new Promise((resolve) => {
    if (!video.videoWidth) return resolve(null);
    const send = liteMode ? SEND_LITE : SEND_FULL;
    const scale = send.width / video.videoWidth;
    grabCanvas.width = send.width;
    grabCanvas.height = Math.round(video.videoHeight * scale);
    grabCtx.drawImage(video, 0, 0, grabCanvas.width, grabCanvas.height);
    grabCanvas.toBlob((b) => resolve(b), "image/jpeg", send.quality);
  });
}

function frameForm(blob) {
  const f = new FormData();
  f.append("file", blob, "frame.jpg");
  return f;
}

// ── Cámara compartida (se inicia una sola vez) ───────────────────────────────
let cameraReady = false;
let cameraStarting = null;

function ensureCamera() {
  if (cameraReady) {
    // El video se pausa al salir de las vistas con cámara: reanudar.
    if (video.paused) video.play().catch(() => {});
    return Promise.resolve();
  }
  if (cameraStarting) return cameraStarting;
  cameraStarting = (async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      });
      video.srcObject = stream;
      await new Promise((res) => (video.onloadedmetadata = res));
      await video.play();
      cameraReady = true;
      if (window.LescoLandmarks)
        window.LescoLandmarks.init(video, landmarksCanvas);
    } catch (err) {
      statusTextEl.textContent = "No se pudo acceder a la cámara";
      console.error(err);
    }
  })();
  return cameraStarting;
}

// ── Router ────────────────────────────────────────────────────────────────────
function setView(name) {
  currentView = name;
  document.body.dataset.view = name;
  document.querySelectorAll(".view").forEach((v) => {
    v.classList.toggle("hidden", v.dataset.view !== name);
  });

  const isCamera = name === "recognize" || name === "collect";
  stage.classList.toggle("hidden", !isCamera);
  backBtn.classList.toggle("hidden", name === "menu");

  // El overlay de landmarks (MediaPipe en el navegador) y el <video> solo
  // trabajan cuando la cámara está en pantalla; fuera de ahí es CPU tirada.
  if (window.LescoLandmarks) window.LescoLandmarks.setActive(isCamera);
  if (!isCamera && cameraReady && !video.paused) video.pause();
  lowFpsStreak = 0;

  // Detener loops activos
  recognizeRunning = false;
  if (streamRecognizer) streamRecognizer.stop();
  collecting = false;
  countdownEl.classList.add("hidden");
  if (name !== "recognize") setTheme("waiting");

  if (name === "recognize") {
    statusTextEl.textContent = "Iniciando…";
    ensureCamera().then(startRecognize);
  } else if (name === "collect") {
    resetCollectUI();
    ensureCamera();
    loadDatasetInto(collectCountsEl, true);
  } else if (name === "train") {
    loadTrainSummary();
    pollTrain(); // refleja un entrenamiento en curso si lo hay
  } else if (name === "inspect") {
    loadInspect();
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// RECONOCER
// ═════════════════════════════════════════════════════════════════════════════
let recognizeRunning = false;
let streamRecognizer = null;

function startRecognize() {
  if (recognizeRunning) return;
  recognizeRunning = true;
  getStreamRecognizer().start();
}

function getStreamRecognizer() {
  if (!streamRecognizer) {
    streamRecognizer = new WebSocketRecognizer({
      video,
      targetFps: TARGET_FPS,
      getSendSettings: () => (liteMode ? SEND_LITE : SEND_FULL),
      onStatus: handleStreamStatus,
      onResult: handleResponse,
      onError: () => {
        if (currentView === "recognize")
          statusTextEl.textContent = "Sin conexión con el servidor";
      },
    });
  }
  return streamRecognizer;
}

function handleStreamStatus(update) {
  if (!recognizeRunning || currentView !== "recognize") return;
  if (update.state === "CONNECTED" && !acceptedActive)
    statusTextEl.textContent = "Esperando seña";
  if (update.state === "RECONNECTING")
    statusTextEl.textContent = "Reconectando…";
}

function handleResponse(data) {
  if (!data) return;
  hud.state = data.state || "WAITING";
  hud.velocity = data.velocity || 0;
  hud.motionThreshold = data.motion_threshold || hud.motionThreshold;

  if (data.error) {
    statusTextEl.textContent = "Modelo no cargado";
    return;
  }

  const result = data.result || {};
  const translation = data.translation || {};
  const info = data.info;
  const rejectedDetection =
    !result.accepted &&
    hud.state !== "SIGNING" &&
    info &&
    info.accepted === false &&
    info.label;
  renderTranslation(translation.text || "");
  hud.accepted = !!result.accepted;

  if (hud.accepted) setTheme("accepted");
  else if (rejectedDetection) setTheme("rejected");
  else if (hud.state === "SIGNING") setTheme("signing");
  else setTheme("waiting");

  if (hud.accepted) {
    statusTextEl.textContent = "Seña reconocida";
  } else if (rejectedDetection) {
    statusTextEl.textContent = rejectedStatusText(info);
  } else if (hud.state === "SIGNING") {
    statusTextEl.textContent = "Capturando seña…";
  } else if (info && info.accepted === false && info.reason) {
    const near = info.label ? ` · ${info.label} ${info.confidence}%` : "";
    statusTextEl.textContent = "No reconocida: " + info.reason + near;
  } else {
    statusTextEl.textContent = "Esperando seña";
  }
  motionLabelEl.textContent =
    hud.state === "SIGNING" ? "CAPTURANDO" : "EN ESPERA";

  if (hud.accepted) {
    showPrediction(result);
    if (!acceptedActive) {
      lastRejectedKey = "";
      addToHistory(result.label);
      speakWord(result.label);
      burstParticles();
      predictionBox.classList.remove("pop");
      void predictionBox.offsetWidth;
      predictionBox.classList.add("pop");
      acceptedActive = true;
    }
  } else if (rejectedDetection) {
    acceptedActive = false;
    showPrediction(info);
    const rejectedKey = `${info.label}:${info.confidence}:${info.reason || ""}`;
    if (rejectedKey !== lastRejectedKey) {
      predictionBox.classList.remove("pop");
      void predictionBox.offsetWidth;
      predictionBox.classList.add("pop");
      lastRejectedKey = rejectedKey;
    }
  } else {
    acceptedActive = false;
    lastRejectedKey = "";
    if (hud.state !== "SIGNING") clearPrediction();
  }
}

function showPrediction(result) {
  predictionEl.textContent = result.label || "---";
  confidenceEl.textContent = (result.confidence || 0) + "%";
  renderBars(result.candidates || []);
}

function clearPrediction() {
  predictionEl.textContent = "---";
  confidenceEl.textContent = "0%";
  barsEl.innerHTML = "";
}

function rejectedStatusText(info) {
  const pct = info.confidence || 0;
  const reason = info.reason ? `: ${info.reason}` : "";
  return `Se detectó ${info.label} ${pct}%, pero está fuera del umbral de aceptabilidad${reason}`;
}

function renderTranslation(text) {
  translationTextEl.textContent = text || "";
}

function renderBars(candidates) {
  barsEl.innerHTML = "";
  candidates.slice(0, 3).forEach(([label, prob]) => {
    const pct = Math.round(prob * 100);
    const row = document.createElement("div");
    row.className = "cbar";
    row.innerHTML = `
            <span class="cbar__label">${label}</span>
            <span class="cbar__track"><span class="cbar__fill"></span></span>
            <span class="cbar__pct">${pct}%</span>`;
    barsEl.appendChild(row);
    const fill = row.querySelector(".cbar__fill");
    void fill.offsetWidth;
    fill.style.width = pct + "%";
  });
}

function addToHistory(label) {
  if (!label) return;
  const empty = historyEl.querySelector(".history__empty");
  if (empty) empty.remove();
  const time = new Date().toLocaleTimeString("es", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const li = document.createElement("li");
  li.innerHTML = `<span class="h-label">${label}</span><span class="h-time">${time}</span>`;
  historyEl.prepend(li);
  while (historyEl.children.length > HISTORY_MAX)
    historyEl.removeChild(historyEl.lastChild);
}

function setTheme(name) {
  const cls = "state-" + name;
  if (document.body.classList.contains(cls)) return;
  document.body.classList.remove(
    "state-waiting",
    "state-signing",
    "state-accepted",
    "state-rejected",
  );
  document.body.classList.add(cls);
  refreshAccentCache();
  if (window.LescoLandmarks) window.LescoLandmarks.refreshAccent();
}

// ── Helpers de color ──────────────────────────────────────────────────────────
// getComputedStyle por frame fuerza recálculo de estilos (caro en equipos
// lentos): el acento se cachea y se refresca solo al cambiar de tema.
let accentCache = { rgb: "0, 229, 255", hex: "#00e5ff" };
function refreshAccentCache() {
  const cs = getComputedStyle(document.body);
  accentCache = {
    rgb: cs.getPropertyValue("--accent-rgb").trim() || "0, 229, 255",
    hex: cs.getPropertyValue("--accent").trim() || "#00e5ff",
  };
}
function accentRGB() {
  return accentCache.rgb;
}
function accentHex() {
  return accentCache.hex;
}

// ── Partículas ────────────────────────────────────────────────────────────────
const pCtx = particlesCanvas.getContext("2d");
let particles = [];

function resizeCanvases() {
  particlesCanvas.width = window.innerWidth;
  particlesCanvas.height = window.innerHeight;
}
window.addEventListener("resize", resizeCanvases);
resizeCanvases();

function spawnParticle(burst = false) {
  const w = particlesCanvas.width,
    h = particlesCanvas.height;
  if (burst) {
    const angle = Math.random() * Math.PI * 2;
    const speed = 2 + Math.random() * 4;
    return {
      x: w / 2,
      y: h - 140,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      r: 1.5 + Math.random() * 2.5,
      life: 1,
      decay: 0.012 + Math.random() * 0.02,
    };
  }
  return {
    x: Math.random() * w,
    y: h + 10,
    vx: (Math.random() - 0.5) * 0.4,
    vy: -(0.3 + Math.random() * 0.8),
    r: 0.8 + Math.random() * 1.8,
    life: 1,
    decay: 0.0,
  };
}

for (let i = 0; i < 70; i++) {
  const p = spawnParticle();
  p.y = Math.random() * particlesCanvas.height;
  particles.push(p);
}

function burstParticles() {
  for (let i = 0; i < 40; i++) particles.push(spawnParticle(true));
}

function updateParticles() {
  const w = particlesCanvas.width,
    h = particlesCanvas.height;
  pCtx.clearRect(0, 0, w, h);
  const rgb = accentRGB();
  const energy = hud.state === "SIGNING" ? 1.8 : 1;
  const next = [];
  for (const p of particles) {
    p.x += p.vx * energy;
    p.y += p.vy * energy;
    if (p.decay) p.life -= p.decay;
    const onScreen = p.x > -10 && p.x < w + 10 && p.y > -10 && p.y < h + 10;
    if (p.life > 0 && onScreen) {
      pCtx.beginPath();
      pCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      pCtx.fillStyle = `rgba(${rgb}, ${0.5 * p.life})`;
      pCtx.fill();
      next.push(p);
    } else if (!p.decay) {
      next.push(spawnParticle());
    }
  }
  particles = next;
  while (particles.length < 70) particles.push(spawnParticle());
}

// ── Radar ───────────────────────────────────────────────────────────────────
const rCtx = radarCanvas.getContext("2d");
let sweep = 0;

function drawRadar() {
  const w = radarCanvas.width,
    h = radarCanvas.height;
  const cx = w / 2,
    cy = h / 2,
    R = w / 2 - 6;
  const rgb = accentRGB(),
    hex = accentHex();
  rCtx.clearRect(0, 0, w, h);

  rCtx.strokeStyle = `rgba(${rgb}, 0.22)`;
  rCtx.lineWidth = 1;
  for (let i = 1; i <= 3; i++) {
    rCtx.beginPath();
    rCtx.arc(cx, cy, (R * i) / 3, 0, Math.PI * 2);
    rCtx.stroke();
  }
  rCtx.beginPath();
  rCtx.moveTo(cx - R, cy);
  rCtx.lineTo(cx + R, cy);
  rCtx.moveTo(cx, cy - R);
  rCtx.lineTo(cx, cy + R);
  rCtx.stroke();

  const ratio = Math.max(
    0,
    Math.min(1, hud.velocity / ((hud.motionThreshold || 0.03) * 3)),
  );
  rCtx.beginPath();
  rCtx.arc(cx, cy, R * ratio, 0, Math.PI * 2);
  rCtx.fillStyle = `rgba(${rgb}, 0.18)`;
  rCtx.fill();
  rCtx.lineWidth = 2;
  rCtx.strokeStyle = hex;
  // shadowBlur es de lo más caro de canvas 2D: solo en modo completo.
  if (!liteMode) {
    rCtx.shadowBlur = 12;
    rCtx.shadowColor = hex;
  }
  rCtx.stroke();
  rCtx.shadowBlur = 0;

  sweep += 0.06;
  const sx = cx + Math.cos(sweep) * R,
    sy = cy + Math.sin(sweep) * R;
  const grad = rCtx.createLinearGradient(cx, cy, sx, sy);
  grad.addColorStop(0, `rgba(${rgb}, 0.85)`);
  grad.addColorStop(1, `rgba(${rgb}, 0)`);
  rCtx.strokeStyle = grad;
  rCtx.lineWidth = 2;
  rCtx.beginPath();
  rCtx.moveTo(cx, cy);
  rCtx.lineTo(sx, sy);
  rCtx.stroke();
}

// ── Loop de render (FPS + radar + partículas) ────────────────────────────────
// El trabajo pesado (radar, partículas) solo corre en las vistas que lo
// muestran; en modo ligero las partículas se apagan del todo.
let frameCount = 0,
  lastFpsTime = performance.now();

function render(now) {
  const isCamera = currentView === "recognize" || currentView === "collect";
  frameCount++;
  if (now - lastFpsTime >= 500) {
    const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
    frameCount = 0;
    lastFpsTime = now;
    if (isCamera) {
      fpsEl.textContent = fps + " FPS";
      trackAutoLite(fps);
    }
  }
  if (currentView === "recognize") drawRadar();
  if (isCamera && !liteMode) updateParticles();
  requestAnimationFrame(render);
}

// ═════════════════════════════════════════════════════════════════════════════
// VOZ (Web Speech API)
//
// Dice en voz alta cada seña aceptada. En Windows usa las mismas voces SAPI
// del sistema (p. ej. Microsoft Sabina/Helena), sin tocar el backend. Es
// prácticamente gratis en CPU: la síntesis la hace el sistema operativo.
// ═════════════════════════════════════════════════════════════════════════════
let voiceOn = localStorage.getItem("lesco-voice") !== "0"; // activada por defecto
let spanishVoice = null;

function pickVoice() {
  if (!window.speechSynthesis) return;
  const voices = speechSynthesis.getVoices();
  // Preferir una voz en español; si no hay, se usa la default del sistema.
  spanishVoice = voices.find((v) => /^es/i.test(v.lang || "")) || null;
}
if (window.speechSynthesis) {
  pickVoice();
  // Las voces cargan de forma asíncrona en Chrome/Edge.
  speechSynthesis.onvoiceschanged = pickVoice;
}

function speakWord(label) {
  if (!voiceOn || !label || !window.speechSynthesis) return;
  const text = label.replace(/_/g, " ").toLowerCase();
  speechSynthesis.cancel(); // si llegan señas seguidas, no encolar
  const u = new SpeechSynthesisUtterance(text);
  if (spanishVoice) u.voice = spanishVoice;
  u.lang = (spanishVoice && spanishVoice.lang) || "es-ES";
  speechSynthesis.speak(u);
}

function updateVoiceBtn() {
  voiceBtn.textContent = voiceOn ? "🔊 VOZ" : "🔇 VOZ";
  voiceBtn.title = voiceOn
    ? "Voz activada: dice cada seña detectada (clic para silenciar)"
    : "Voz silenciada (clic para activar)";
}

// ═════════════════════════════════════════════════════════════════════════════
// MODO LIGERO (laptops de bajos recursos)
//
// Apaga lo puramente cosmético: overlay de landmarks del navegador, partículas,
// glows de canvas y efectos CSS caros (backdrop-filter, scanline), y manda
// frames más chicos al backend. No toca la lógica de reconocimiento.
// ═════════════════════════════════════════════════════════════════════════════
let liteMode = false;
let lowFpsStreak = 0;
let autoLiteAllowed = localStorage.getItem("lesco-lite") === null;

function applyLite(on) {
  liteMode = on;
  document.body.classList.toggle("lite", on);
  if (perfBtn) {
    perfBtn.textContent = on ? "⚡ LIGERO" : "✦ COMPLETO";
    perfBtn.title = on
      ? "Modo ligero activo: efectos apagados para mejor FPS (clic para volver)"
      : "Clic para modo ligero (mejor FPS en equipos modestos)";
  }
  if (window.LescoLandmarks) window.LescoLandmarks.setLite(on);
  if (on) pCtx.clearRect(0, 0, particlesCanvas.width, particlesCanvas.height);
}

// Si el FPS se mantiene bajo unos segundos en una vista con cámara, se activa
// solo (salvo que el usuario ya haya elegido un modo manualmente).
function trackAutoLite(fps) {
  if (liteMode || !autoLiteAllowed) return;
  lowFpsStreak = fps < 20 ? lowFpsStreak + 1 : 0;
  if (lowFpsStreak >= 8) {
    // ~4 s sostenidos por debajo de 20 FPS
    autoLiteAllowed = false;
    applyLite(true);
    toast("⚡ Modo ligero activado (FPS bajo)", "ok");
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// GRABAR
// ═════════════════════════════════════════════════════════════════════════════
let collecting = false;

function resetCollectUI() {
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  discardBtn.disabled = true;
  framesEl.textContent = "0";
  detectedEl.textContent = "0";
}

function runCountdown(seconds) {
  return new Promise((resolve) => {
    countdownEl.classList.remove("rec", "hidden");
    let c = seconds;
    countdownEl.textContent = c;
    const iv = setInterval(() => {
      c--;
      if (c <= 0) {
        clearInterval(iv);
        resolve();
      } else countdownEl.textContent = c;
    }, 1000);
  });
}

async function startRecording() {
  const label = (labelInput.value || "").trim();
  if (!label) {
    toast("Escribí una etiqueta primero", "err");
    return;
  }

  recordBtn.disabled = true;
  await runCountdown(3);
  if (currentView !== "collect") return; // salió de la vista

  await fetch("/collect/start", { method: "POST" });
  collecting = true;
  stopBtn.disabled = false;
  discardBtn.disabled = false;
  countdownEl.textContent = "● REC";
  countdownEl.classList.add("rec");
  countdownEl.classList.remove("hidden");
  collectTick();
}

async function collectTick() {
  if (!collecting) return;
  const t0 = performance.now();
  const blob = await captureBlob();
  if (blob) {
    try {
      const res = await fetch("/collect/frame", {
        method: "POST",
        body: frameForm(blob),
      });
      const d = await res.json();
      framesEl.textContent = d.frames;
      const ratio = d.frames ? Math.round((d.detected / d.frames) * 100) : 0;
      detectedEl.textContent = `${d.detected} (${ratio}%)`;
    } catch (err) {
      /* frame perdido */
    }
  }
  if (!collecting) return;
  const elapsed = performance.now() - t0;
  setTimeout(collectTick, Math.max(0, TARGET_INTERVAL_MS - elapsed));
}

async function finishRecording(save) {
  collecting = false;
  countdownEl.classList.add("hidden");
  countdownEl.classList.remove("rec");
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  discardBtn.disabled = true;

  if (save) {
    const label = (labelInput.value || "").trim();
    const f = new FormData();
    f.append("label", label);
    try {
      const res = await fetch("/collect/save", { method: "POST", body: f });
      const d = await res.json();
      if (d.ok) {
        toast(`✓ ${d.label} guardada (${d.total} muestras)`, "ok");
        renderCounts(d.per_label);
        refreshLabels(d.per_label);
      } else {
        toast("✗ " + (d.reason || "no se pudo guardar"), "err");
      }
    } catch (err) {
      toast("✗ error al guardar", "err");
    }
  } else {
    await fetch("/collect/discard", { method: "POST" });
    toast("Toma descartada", "err");
  }
  framesEl.textContent = "0";
  detectedEl.textContent = "0";
}

function isTypingTarget(target) {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target?.isContentEditable
  );
}

function handleCollectShortcut(event) {
  if (currentView !== "collect" || event.repeat || isTypingTarget(event.target))
    return;
  if (event.ctrlKey || event.metaKey || event.altKey) return;

  const key = event.key.toLowerCase();
  if (key === "a" && !recordBtn.disabled) {
    event.preventDefault();
    startRecording();
  } else if (key === "s" && !stopBtn.disabled) {
    event.preventDefault();
    finishRecording(true);
  } else if (key === "d" && !discardBtn.disabled) {
    event.preventDefault();
    finishRecording(false);
  }
}

let toastTimer = null;
function toast(msg, kind) {
  toastEl.textContent = msg;
  toastEl.className = "toast " + (kind || "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.add("hidden"), 2800);
}

function renderCounts(perLabel) {
  const entries = Object.entries(perLabel || {});
  if (!entries.length) {
    collectCountsEl.innerHTML = '<li class="history__empty">Sin muestras</li>';
    return;
  }
  collectCountsEl.innerHTML = entries
    .sort((a, b) => b[1] - a[1])
    .map(
      ([l, c]) => `<li><span>${l}</span><span class="h-time">${c}</span></li>`,
    )
    .join("");
}

function refreshLabels(perLabel) {
  labelList.innerHTML = Object.keys(perLabel || {})
    .map((l) => `<option value="${l}">`)
    .join("");
}

// ═════════════════════════════════════════════════════════════════════════════
// DATASET (compartido por Grabar, Entrenar, Inspeccionar)
// ═════════════════════════════════════════════════════════════════════════════
async function fetchDataset() {
  try {
    const res = await fetch("/api/dataset");
    return await res.json();
  } catch (err) {
    return null;
  }
}

function summaryText(d) {
  if (!d) return "No se pudo leer el dataset.";
  if (!d.exists || !d.total_samples)
    return "Dataset vacío: grabá muestras primero.";
  const labels = Object.keys(d.per_label || {}).length;
  return `Total: ${d.total_samples} muestras · ${labels} etiquetas`;
}

async function loadDatasetInto(listEl, alsoLabels) {
  const d = await fetchDataset();
  const per = (d && d.per_label) || {};
  renderCounts(per);
  if (alsoLabels) refreshLabels(per);
}

// ═════════════════════════════════════════════════════════════════════════════
// ENTRENAR
// ═════════════════════════════════════════════════════════════════════════════
async function loadTrainSummary() {
  trainSummaryEl.textContent = "Cargando dataset…";
  trainSummaryEl.textContent = summaryText(await fetchDataset());
}

async function startTraining() {
  trainStartBtn.disabled = true;
  trainStatusEl.className = "train-status";
  trainStatusEl.textContent = "Entrenando…";
  trainMetricsEl.innerHTML = "";
  try {
    await fetch("/train", { method: "POST" });
  } catch (err) {
    trainStatusEl.className = "train-status error";
    trainStatusEl.textContent = "✗ no se pudo iniciar";
    trainStartBtn.disabled = false;
    return;
  }
  pollTrain();
}

async function pollTrain() {
  let s;
  try {
    s = await (await fetch("/train/status")).json();
  } catch (err) {
    return;
  }

  if (s.status === "running") {
    trainStartBtn.disabled = true;
    trainStatusEl.className = "train-status";
    trainStatusEl.textContent = s.message || "Entrenando…";
    setTimeout(pollTrain, 1500);
    return;
  }
  trainStartBtn.disabled = false;
  if (s.status === "done") {
    trainStatusEl.className = "train-status";
    trainStatusEl.textContent = "✓ " + s.message;
    renderMetrics(s.metrics);
  } else if (s.status === "error") {
    trainStatusEl.className = "train-status error";
    trainStatusEl.textContent = "✗ " + s.message;
  }
}

function renderMetrics(m) {
  if (!m) {
    trainMetricsEl.innerHTML = "";
    return;
  }
  const val = m.val_acc == null ? "—" : Math.round(m.val_acc * 100) + "%";
  trainMetricsEl.innerHTML = `
        <div class="metric"><b>${Math.round(m.train_acc * 100)}%</b><span>Precisión entrenamiento</span></div>
        <div class="metric"><b>${val}</b><span>Precisión validación</span></div>
        <div class="metric"><b>${m.train_samples}</b><span>Muestras de entrenamiento</span></div>
        <div class="metric"><b>${m.labels.length}</b><span>${m.labels.join(", ")}</span></div>`;
}

// ═════════════════════════════════════════════════════════════════════════════
// INSPECCIONAR
// ═════════════════════════════════════════════════════════════════════════════
async function loadInspect() {
  inspectSummaryEl.textContent = "Cargando…";
  const d = await fetchDataset();
  inspectSummaryEl.textContent = summaryText(d);
  renderTable((d && d.per_label) || {});
}

function renderTable(perLabel) {
  const entries = Object.entries(perLabel || {});
  if (!entries.length) {
    inspectBodyEl.innerHTML =
      '<tr><td colspan="3" style="color:var(--text-dim)">Sin muestras</td></tr>';
    return;
  }
  const max = Math.max(...entries.map((e) => e[1]));
  inspectBodyEl.innerHTML = entries
    .sort((a, b) => b[1] - a[1])
    .map(
      ([l, c]) =>
        `<tr><td>${l}</td><td><b>${c}</b></td>
             <td><div class="bar-cell"><span style="width:${(c / max) * 100}%"></span></div></td></tr>`,
    )
    .join("");
}

// ═════════════════════════════════════════════════════════════════════════════
// WIRING + ARRANQUE
// ═════════════════════════════════════════════════════════════════════════════
document.querySelectorAll(".menu-card").forEach((card) => {
  card.addEventListener("click", () => setView(card.dataset.go));
});
backBtn.addEventListener("click", () => setView("menu"));
recordBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", () => finishRecording(true));
discardBtn.addEventListener("click", () => finishRecording(false));
window.addEventListener("keydown", handleCollectShortcut);
trainStartBtn.addEventListener("click", startTraining);
inspectRefreshBtn.addEventListener("click", loadInspect);
perfBtn.addEventListener("click", () => {
  autoLiteAllowed = false; // elección manual: no volver a auto-activar
  applyLite(!liteMode);
  localStorage.setItem("lesco-lite", liteMode ? "1" : "0");
});
voiceBtn.addEventListener("click", () => {
  voiceOn = !voiceOn;
  localStorage.setItem("lesco-voice", voiceOn ? "1" : "0");
  updateVoiceBtn();
  // Feedback inmediato; además el clic desbloquea el audio del navegador.
  if (voiceOn) speakWord("voz activada");
  else if (window.speechSynthesis) speechSynthesis.cancel();
});

refreshAccentCache();
updateVoiceBtn();
applyLite(localStorage.getItem("lesco-lite") === "1");
requestAnimationFrame(render);
setView("menu");
