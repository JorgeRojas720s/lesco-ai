const encodeCanvas = new OffscreenCanvas(1, 1);
const encodeCtx = encodeCanvas.getContext("2d", { alpha: false });
const motionCanvas = new OffscreenCanvas(96, 54);
const motionCtx = motionCanvas.getContext("2d", {
    alpha: false,
    willReadFrequently: true,
});

let previousGray = null;

function resizeCanvas(canvas, width, height) {
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
}

function motionScore(bitmap) {
    const w = motionCanvas.width;
    const h = Math.max(1, Math.round(bitmap.height * (w / bitmap.width)));
    resizeCanvas(motionCanvas, w, h);
    motionCtx.drawImage(bitmap, 0, 0, w, h);

    const pixels = motionCtx.getImageData(0, 0, w, h).data;
    const gray = new Uint8Array(w * h);
    let diff = 0;

    for (let i = 0, j = 0; i < pixels.length; i += 4, j++) {
        gray[j] = (pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114) | 0;
        if (previousGray) diff += Math.abs(gray[j] - previousGray[j]);
    }

    previousGray = gray;
    return previousGray ? diff / gray.length : 0;
}

async function encodeFrame(bitmap, width, quality, type) {
    const height = Math.max(1, Math.round(bitmap.height * (width / bitmap.width)));
    resizeCanvas(encodeCanvas, width, height);
    encodeCtx.drawImage(bitmap, 0, 0, width, height);
    const blob = await encodeCanvas.convertToBlob({ type, quality });
    return {
        buffer: await blob.arrayBuffer(),
        mime: blob.type || type,
        width,
        height,
    };
}

self.onmessage = async (event) => {
    const msg = event.data;
    if (msg.type === "reset") {
        previousGray = null;
        return;
    }
    if (msg.type !== "frame") return;

    const bitmap = msg.bitmap;
    try {
        const motion = motionScore(bitmap);
        const encoded = await encodeFrame(bitmap, msg.width, msg.quality, msg.mime);
        self.postMessage({
            type: "frame",
            id: msg.id,
            generation: msg.generation,
            motion,
            ...encoded,
        }, [encoded.buffer]);
    } catch (err) {
        self.postMessage({
            type: "error",
            id: msg.id,
            generation: msg.generation,
            message: err?.message || "worker frame error",
        });
    } finally {
        bitmap.close();
    }
};
