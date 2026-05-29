/* ─────────────────────────────────────────────────────────────────────────
   app.js · Cliente del HUD LESCO-AI

   - Captura frames de la cámara y los envía a POST /predict (mismo origen).
   - Pinta el HUD: estado, FPS, radar de movimiento, partículas, barras de
     confianza, historial de señas y modo oscuro dinámico.
   - El overlay de landmarks lo maneja landmarks.js (MediaPipe en navegador).

   No contiene lógica de reconocimiento: solo consume la respuesta del backend.
   ───────────────────────────────────────────────────────────────────────── */

// ── Referencias DOM ─────────────────────────────────────────────────────────
const video = document.getElementById("video");
const landmarksCanvas = document.getElementById("landmarks");
const particlesCanvas = document.getElementById("particles");
const radarCanvas = document.getElementById("radar");

const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const barsEl = document.getElementById("confidence-bars");
const statusTextEl = document.getElementById("status-text");
const fpsEl = document.getElementById("fps");
const motionLabelEl = document.getElementById("motion-label");
const historyEl = document.getElementById("history");
const predictionBox = document.getElementById("prediction-box");

const PREDICT_URL = "/predict";
const FRAME_INTERVAL_MS = 120;
const HISTORY_MAX = 8;

// ── Estado compartido entre el loop de red y el de render ───────────────────
const hud = {
    state: "WAITING", // WAITING | SIGNING
    velocity: 0,
    motionThreshold: 0.03,
    accepted: false,
};

let acceptedActive = false; // evita duplicar entradas de historial durante el hold

// ── Captura de cámara y envío de frames ─────────────────────────────────────
const grabCanvas = document.createElement("canvas");
const grabCtx = grabCanvas.getContext("2d");

async function initCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
    });
    video.srcObject = stream;
    await new Promise((res) => (video.onloadedmetadata = res));
    await video.play();
}

async function sendFrame() {
    if (!video.videoWidth) {
        setTimeout(sendFrame, FRAME_INTERVAL_MS);
        return;
    }

    grabCanvas.width = video.videoWidth;
    grabCanvas.height = video.videoHeight;
    grabCtx.drawImage(video, 0, 0);

    grabCanvas.toBlob(
        async (blob) => {
            if (blob) {
                const formData = new FormData();
                formData.append("file", blob, "frame.jpg");
                try {
                    const res = await fetch(PREDICT_URL, { method: "POST", body: formData });
                    handleResponse(await res.json());
                } catch (err) {
                    statusTextEl.textContent = "Sin conexión con el servidor";
                }
            }
            setTimeout(sendFrame, FRAME_INTERVAL_MS);
        },
        "image/jpeg",
        0.7
    );
}

// ── Manejo de la respuesta del backend ──────────────────────────────────────
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
    hud.accepted = !!result.accepted;

    // Modo oscuro dinámico
    if (hud.accepted) setTheme("accepted");
    else if (hud.state === "SIGNING") setTheme("signing");
    else setTheme("waiting");

    // Texto de estado / radar
    statusTextEl.textContent = hud.accepted
        ? "Seña reconocida"
        : hud.state === "SIGNING"
        ? "Capturando seña…"
        : "Esperando seña";
    motionLabelEl.textContent = hud.state === "SIGNING" ? "CAPTURANDO" : "EN ESPERA";

    if (hud.accepted) {
        showPrediction(result);
        if (!acceptedActive) {
            addToHistory(result.label);
            burstParticles();
            predictionBox.classList.remove("pop");
            void predictionBox.offsetWidth; // reinicia animación
            predictionBox.classList.add("pop");
            acceptedActive = true;
        }
    } else {
        acceptedActive = false;
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
        // fuerza un reflow con width:0 y luego anima hasta pct
        const fill = row.querySelector(".cbar__fill");
        void fill.offsetWidth;
        fill.style.width = pct + "%";
    });
}

// ── Historial ────────────────────────────────────────────────────────────────
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

    while (historyEl.children.length > HISTORY_MAX) {
        historyEl.removeChild(historyEl.lastChild);
    }
}

// ── Modo oscuro dinámico ─────────────────────────────────────────────────────
function setTheme(name) {
    const cls = "state-" + name;
    if (document.body.classList.contains(cls)) return;
    document.body.classList.remove("state-waiting", "state-signing", "state-accepted");
    document.body.classList.add(cls);
}

// ── Helpers de color (leen las variables CSS del tema activo) ────────────────
function cssVar(name) {
    return getComputedStyle(document.body).getPropertyValue(name).trim();
}
function accentRGB() {
    return cssVar("--accent-rgb") || "0, 229, 255";
}
function accentHex() {
    return cssVar("--accent") || "#00e5ff";
}

// ── Partículas ───────────────────────────────────────────────────────────────
const pCtx = particlesCanvas.getContext("2d");
let particles = [];

function resizeCanvases() {
    particlesCanvas.width = window.innerWidth;
    particlesCanvas.height = window.innerHeight;
}
window.addEventListener("resize", resizeCanvases);
resizeCanvases();

function spawnParticle(burst = false) {
    const w = particlesCanvas.width;
    const h = particlesCanvas.height;
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

// campo ambiental de partículas
for (let i = 0; i < 70; i++) {
    const p = spawnParticle();
    p.y = Math.random() * particlesCanvas.height;
    particles.push(p);
}

function burstParticles() {
    for (let i = 0; i < 40; i++) particles.push(spawnParticle(true));
}

function updateParticles() {
    const w = particlesCanvas.width;
    const h = particlesCanvas.height;
    pCtx.clearRect(0, 0, w, h);

    const rgb = accentRGB();
    // velocidad sube cuando hay movimiento
    const energy = hud.state === "SIGNING" ? 1.8 : 1;

    const next = [];
    for (const p of particles) {
        p.x += p.vx * energy;
        p.y += p.vy * energy;
        if (p.decay) p.life -= p.decay;

        const onScreen = p.x > -10 && p.x < w + 10 && p.y > -10 && p.y < h + 10;
        const alive = p.life > 0 && onScreen;

        if (alive) {
            pCtx.beginPath();
            pCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            pCtx.fillStyle = `rgba(${rgb}, ${0.5 * p.life})`;
            pCtx.fill();
            next.push(p);
        } else if (!p.decay) {
            // partícula ambiental: reciclar
            next.push(spawnParticle());
        }
    }
    particles = next;

    // mantener una densidad mínima
    while (particles.length < 70) particles.push(spawnParticle());
}

// ── Radar de movimiento ──────────────────────────────────────────────────────
const rCtx = radarCanvas.getContext("2d");
let sweep = 0;

function drawRadar() {
    const w = radarCanvas.width;
    const h = radarCanvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const R = w / 2 - 6;
    const rgb = accentRGB();
    const hex = accentHex();

    rCtx.clearRect(0, 0, w, h);

    // anillos y cruz
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

    // disco proporcional a la velocidad (misma fórmula que el HUD cv2)
    const ratio = Math.max(
        0,
        Math.min(1, hud.velocity / ((hud.motionThreshold || 0.03) * 3))
    );
    rCtx.beginPath();
    rCtx.arc(cx, cy, R * ratio, 0, Math.PI * 2);
    rCtx.fillStyle = `rgba(${rgb}, 0.18)`;
    rCtx.fill();
    rCtx.lineWidth = 2;
    rCtx.strokeStyle = hex;
    rCtx.shadowBlur = 12;
    rCtx.shadowColor = hex;
    rCtx.stroke();
    rCtx.shadowBlur = 0;

    // barrido giratorio
    sweep += 0.06;
    const sx = cx + Math.cos(sweep) * R;
    const sy = cy + Math.sin(sweep) * R;
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

// ── Loop de render unificado (FPS + radar + partículas) ──────────────────────
let frameCount = 0;
let lastFpsTime = performance.now();

function render(now) {
    frameCount++;
    if (now - lastFpsTime >= 500) {
        const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
        fpsEl.textContent = fps + " FPS";
        frameCount = 0;
        lastFpsTime = now;
    }
    drawRadar();
    updateParticles();
    requestAnimationFrame(render);
}

// ── Arranque ─────────────────────────────────────────────────────────────────
async function start() {
    // El HUD ambiental (radar, partículas, FPS) corre siempre, aunque la
    // cámara falle, para no quedar congelado.
    requestAnimationFrame(render);

    try {
        await initCamera();
    } catch (err) {
        statusTextEl.textContent = "No se pudo acceder a la cámara";
        console.error(err);
        return;
    }
    if (window.LescoLandmarks) {
        window.LescoLandmarks.init(video, landmarksCanvas);
    }
    sendFrame();
}

start();
