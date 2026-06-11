const DEFAULTS = {
  targetFps: 30,
  path: "/ws/predict",
  mime: "image/jpeg",
  maxWorkerQueue: 4,
  maxSocketBufferBytes: 4_000_000,
};

export class WebSocketRecognizer {
  constructor(options) {
    this.video = options.video;
    this.getSendSettings = options.getSendSettings;
    this.onStatus = options.onStatus || (() => {});
    this.onResult = options.onResult || (() => {});
    this.onError = options.onError || (() => {});
    this.config = { ...DEFAULTS, ...options };

    this.worker = new Worker(new URL("./frame_worker.js", import.meta.url));
    this.worker.onmessage = (event) => this.handleWorkerMessage(event.data);

    this.running = false;
    this.socket = null;
    this.frameId = 0;
    this.generation = 0;
    this.inWorker = 0;
    this.loopTimer = null;
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.generation += 1;
    this.inWorker = 0;
    this.worker.postMessage({ type: "reset" });
    this.connect();
    this.scheduleNext(0);
  }

  stop() {
    this.running = false;
    if (this.loopTimer) clearTimeout(this.loopTimer);
    this.loopTimer = null;
    this.generation += 1;
    this.inWorker = 0;
    this.worker.postMessage({ type: "reset" });
    if (this.socket) {
      this.socket.onclose = null;
      this.socket.close();
      this.socket = null;
    }
  }

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}${this.config.path}`;
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    this.socket = socket;

    socket.onopen = () => this.onStatus({ state: "CONNECTED" });
    socket.onmessage = (event) => {
      try {
        this.onResult(JSON.parse(event.data));
      } catch (err) {
        this.onError(err);
      }
    };
    socket.onerror = () => this.onError(new Error("websocket error"));
    socket.onclose = () => {
      if (!this.running) return;
      this.onStatus({ state: "RECONNECTING" });
      setTimeout(() => {
        if (this.running) this.connect();
      }, 500);
    };
  }

  scheduleNext(delay) {
    if (!this.running) return;
    this.loopTimer = setTimeout(() => this.captureTick(), delay);
  }

  async captureTick() {
    if (!this.running) return;
    const startedAt = performance.now();

    if (
      this.video.readyState >= 2 &&
      this.inWorker < this.config.maxWorkerQueue &&
      this.canSend() &&
      "createImageBitmap" in window
    ) {
      try {
        const bitmap = await createImageBitmap(this.video);
        const settings = this.getSendSettings();
        this.inWorker += 1;
        this.worker.postMessage(
          {
            type: "frame",
            id: ++this.frameId,
            generation: this.generation,
            bitmap,
            width: settings.width,
            quality: settings.quality,
            mime: this.config.mime,
          },
          [bitmap],
        );
      } catch (err) {
        this.onError(err);
      }
    }

    const elapsed = performance.now() - startedAt;
    this.scheduleNext(Math.max(0, 1000 / this.config.targetFps - elapsed));
  }

  handleWorkerMessage(message) {
    if (message.generation !== this.generation) return;
    this.inWorker = Math.max(0, this.inWorker - 1);

    if (message.type === "error") {
      this.onError(new Error(message.message));
      return;
    }
    if (message.type !== "frame" || !this.running || !this.canSend()) return;
    this.socket.send(message.buffer);
  }

  canSend() {
    return (
      this.socket &&
      this.socket.readyState === WebSocket.OPEN &&
      this.socket.bufferedAmount <= this.config.maxSocketBufferBytes
    );
  }
}
