const DEFAULTS = {
    targetFps: 30,
    endpoint: "/predict/batch",
    startMotion: 4.0,
    stopMotion: 2.2,
    silenceFrames: 10,
    minFrames: 12,
    maxFrames: 180,
    preRollFrames: 6,
    maxWorkerQueue: 18,
    mime: "image/jpeg",
};

export class BatchRecognizer {
    constructor(options) {
        this.video = options.video;
        this.getSendSettings = options.getSendSettings;
        this.onState = options.onState || (() => {});
        this.onResult = options.onResult || (() => {});
        this.onError = options.onError || (() => {});
        this.config = { ...DEFAULTS, ...options };

        this.worker = new Worker(new URL("./frame_worker.js", import.meta.url));
        this.worker.onmessage = (event) => this.handleWorkerMessage(event.data);

        this.running = false;
        this.state = "WAITING";
        this.frameId = 0;
        this.inWorker = 0;
        this.loopTimer = null;
        this.generation = 0;
        this.preRoll = [];
        this.batch = [];
        this.silence = 0;
        this.pendingSends = Promise.resolve();
    }

    start() {
        if (this.running) return;
        this.running = true;
        this.resetCapture();
        this.scheduleNext(0);
    }

    stop() {
        this.running = false;
        if (this.loopTimer) clearTimeout(this.loopTimer);
        this.loopTimer = null;
        this.resetCapture();
    }

    resetCapture() {
        this.generation += 1;
        this.state = "WAITING";
        this.preRoll = [];
        this.batch = [];
        this.silence = 0;
        this.inWorker = 0;
        this.worker.postMessage({ type: "reset" });
        this.emitState();
    }

    scheduleNext(delay) {
        if (!this.running) return;
        this.loopTimer = setTimeout(() => this.captureTick(), delay);
    }

    async captureTick() {
        if (!this.running) return;
        const startedAt = performance.now();

        if (
            this.video.readyState >= 2
            && this.inWorker < this.config.maxWorkerQueue
            && "createImageBitmap" in window
        ) {
            try {
                const bitmap = await createImageBitmap(this.video);
                const settings = this.getSendSettings();
                this.inWorker += 1;
                this.worker.postMessage({
                    type: "frame",
                    id: ++this.frameId,
                    generation: this.generation,
                    bitmap,
                    width: settings.width,
                    quality: settings.quality,
                    mime: this.config.mime,
                }, [bitmap]);
            } catch (err) {
                this.onError(err);
            }
        }

        const elapsed = performance.now() - startedAt;
        this.scheduleNext(Math.max(0, 1000 / this.config.targetFps - elapsed));
    }

    handleWorkerMessage(message) {
        if (message.generation !== undefined && message.generation !== this.generation) return;
        if (message.type === "error") {
            this.inWorker = Math.max(0, this.inWorker - 1);
            this.onError(new Error(message.message));
            return;
        }
        if (message.type !== "frame") return;

        if (!this.running) return;
        this.inWorker = Math.max(0, this.inWorker - 1);
        const frame = {
            buffer: message.buffer,
            mime: message.mime,
            motion: message.motion,
        };

        if (this.state === "WAITING") {
            this.pushPreRoll(frame);
            if (message.motion >= this.config.startMotion) {
                this.state = "SIGNING";
                this.batch = this.preRoll.slice();
                this.preRoll = [];
                this.silence = 0;
            }
        } else if (this.state === "SIGNING") {
            this.batch.push(frame);
            if (message.motion <= this.config.stopMotion) this.silence += 1;
            else this.silence = 0;

            const hasEnoughSilence = (
                this.batch.length >= this.config.minFrames
                && this.silence >= this.config.silenceFrames
            );
            const tooLong = this.batch.length >= this.config.maxFrames;
            if (hasEnoughSilence || tooLong) this.flushBatch();
        }

        this.emitState(message.motion);
    }

    pushPreRoll(frame) {
        this.preRoll.push(frame);
        while (this.preRoll.length > this.config.preRollFrames) this.preRoll.shift();
    }

    flushBatch() {
        const frames = this.batch.slice();
        this.batch = [];
        this.preRoll = [];
        this.silence = 0;
        this.state = "WAITING";
        if (frames.length < this.config.minFrames) return;

        this.pendingSends = this.pendingSends
            .catch(() => undefined)
            .then(() => this.sendBatch(frames));
    }

    async sendBatch(frames) {
        this.onState({
            state: "PROCESSING",
            motion: 0,
            motionThreshold: this.config.startMotion,
            frames: frames.length,
            queued: this.inWorker,
        });

        const form = new FormData();
        frames.forEach((frame, index) => {
            form.append(
                "files",
                new Blob([frame.buffer], { type: frame.mime }),
                `frame-${String(index).padStart(4, "0")}.jpg`,
            );
        });

        try {
            const response = await fetch(this.config.endpoint, {
                method: "POST",
                body: form,
            });
            this.onResult(await response.json());
        } catch (err) {
            this.onError(err);
        }
    }

    emitState(motion = 0) {
        this.onState({
            state: this.state,
            motion,
            motionThreshold: this.config.startMotion,
            frames: this.batch.length,
            queued: this.inWorker,
        });
    }
}
