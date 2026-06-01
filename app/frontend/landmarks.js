/* ─────────────────────────────────────────────────────────────────────────
   landmarks.js · Overlay de landmarks con MediaPipe Hands (solo navegador)

   Dibuja el esqueleto de la(s) mano(s) sobre el <canvas id="landmarks">.
   Es PURAMENTE visual y vive en el navegador: NO interviene en la lógica de
   detección/clasificación del backend Python.

   El video se muestra espejado por CSS (scaleX(-1)); este canvas también se
   espeja por CSS (.mirrored) y se dibuja en coordenadas crudas del video, de
   modo que el esqueleto queda perfectamente alineado con lo que se ve.
   ───────────────────────────────────────────────────────────────────────── */

(function () {
    let hands = null;
    let video, canvas, ctx;
    let running = false;
    let handsPresent = false;

    function accent() {
        return (
            getComputedStyle(document.body).getPropertyValue("--accent").trim() ||
            "#00e5ff"
        );
    }

    function syncSize() {
        if (!video.videoWidth) return;
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }
    }

    function onResults(results) {
        syncSize();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const list = results.multiHandLandmarks || [];
        handsPresent = list.length > 0;
        const color = accent();

        for (const lm of list) {
            if (window.drawConnectors && window.HAND_CONNECTIONS) {
                window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, {
                    color: color,
                    lineWidth: 3,
                });
            }
            if (window.drawLandmarks) {
                window.drawLandmarks(ctx, lm, {
                    color: "#ffffff",
                    fillColor: color,
                    lineWidth: 1,
                    radius: 4,
                });
            }
        }
    }

    async function loop() {
        if (!running) return;
        if (hands && video.readyState >= 2) {
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
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5,
            });
            hands.onResults(onResults);

            running = true;
            loop();
        },

        get handsPresent() {
            return handsPresent;
        },
    };
})();
