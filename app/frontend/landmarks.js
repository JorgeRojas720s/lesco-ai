/* ─────────────────────────────────────────────────────────────────────────
   landmarks.js · Overlay de landmarks con MediaPipe Hands (solo navegador)

   Dibuja el esqueleto de la(s) mano(s) sobre el <canvas id="landmarks">.
   Es PURAMENTE visual y vive en el navegador: NO interviene en la lógica de
   detección/clasificación del backend Python.

   El video se muestra espejado por CSS (scaleX(-1)); este canvas también se
   espeja por CSS (.mirrored) y se dibuja en coordenadas crudas del video, de
   modo que el esqueleto queda perfectamente alineado con lo que se ve.

   Rendimiento: el overlay solo corre en vistas con cámara (setActive) y se
   apaga por completo en modo ligero (setLite), porque en equipos modestos
   esta red compite por CPU con el reconocedor del backend.
   ───────────────────────────────────────────────────────────────────────── */

(function () {
    let hands = null;
    let video, canvas, ctx;
    let running = false;       // init completado, loop vivo
    let active = false;        // true solo en vistas con cámara visible
    let lite = false;          // modo ligero: overlay apagado
    let handsPresent = false;
    let accentColor = "#00e5ff";
    let lastSend = 0;

    // La cámara no pasa de ~30 fps: correr la red al ritmo de rAF (hasta 60)
    // es inferir dos veces el mismo frame. Se limita el envío a 30 fps.
    const SEND_INTERVAL_MS = 1000 / 30;

    // getComputedStyle por frame fuerza recálculo de estilos; el acento se
    // cachea y app.js lo refresca al cambiar de tema.
    function refreshAccent() {
        accentColor =
            getComputedStyle(document.body).getPropertyValue("--accent").trim() ||
            "#00e5ff";
    }

    function syncSize() {
        if (!video.videoWidth) return;
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }
    }

    function clearOverlay() {
        handsPresent = false;
        if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function onResults(results) {
        syncSize();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const list = results.multiHandLandmarks || [];
        handsPresent = list.length > 0;

        for (const lm of list) {
            if (window.drawConnectors && window.HAND_CONNECTIONS) {
                window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, {
                    color: accentColor,
                    lineWidth: 3,
                });
            }
            if (window.drawLandmarks) {
                window.drawLandmarks(ctx, lm, {
                    color: "#ffffff",
                    fillColor: accentColor,
                    lineWidth: 1,
                    radius: 4,
                });
            }
        }
    }

    async function loop() {
        if (!running) return;
        const now = performance.now();
        if (
            active && !lite && hands &&
            video.readyState >= 2 &&
            now - lastSend >= SEND_INTERVAL_MS - 1
        ) {
            lastSend = now;
            try {
                await hands.send({ image: video });
            } catch (err) {
                /* frame ocasional fallido: se ignora */
            }
        }
        requestAnimationFrame(loop);
    }

    window.LescoLandmarks = {
        async init(videoEl, canvasEl) {
            video = videoEl;
            canvas = canvasEl;
            ctx = canvas.getContext("2d");

            if (typeof Hands === "undefined") {
                console.warn("MediaPipe Hands no disponible (CDN). Overlay desactivado.");
                return;
            }

            hands = new Hands({
                locateFile: (f) =>
                    `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${f}`,
            });
            hands.setOptions({
                maxNumHands: 2,
                // 0 = modelo lite: para dibujar el esqueleto se ve igual y
                // cuesta mucho menos CPU (este overlay es solo visual).
                modelComplexity: 0,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5,
            });
            hands.onResults(onResults);

            refreshAccent();
            running = true;
            loop();
        },

        /** Corre el overlay solo cuando la cámara está en pantalla. */
        setActive(value) {
            active = value;
            if (!value) clearOverlay();
        },

        /** Modo ligero: apaga el overlay por completo. */
        setLite(value) {
            lite = value;
            if (value) clearOverlay();
        },

        refreshAccent,

        get handsPresent() {
            return handsPresent;
        },
    };
})();
