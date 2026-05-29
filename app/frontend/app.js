const video = document.getElementById("video");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const statusText = document.getElementById("status");

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

async function initCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: 1280,
            height: 720,
        },
        audio: false,
    });

    video.srcObject = stream;
}

async function sendFrame() {
    if (!video.videoWidth) {
        requestAnimationFrame(sendFrame);
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        try {
            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (data.result.accepted) {
                prediction.innerText = data.result.label;
                confidence.innerText = data.result.confidence + "%";
            }

            statusText.innerText = data.state;

        } catch (error) {
            console.error(error);
        }
    }, "image/jpeg", 0.7);

    setTimeout(sendFrame, 120);
}

async function start() {
    await initCamera();
    sendFrame();
}

start();