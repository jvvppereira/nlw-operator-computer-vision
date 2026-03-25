const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const qualitySlider = document.getElementById('quality-slider');
const qualityValue = document.getElementById('quality-value');
const drawLandmarks = document.getElementById('draw-landmarks');

qualitySlider.oninput = () => {
    qualityValue.innerText = qualitySlider.value;
};

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

// Connect WebSocket (adjusted to support SSL)
const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${protocol}//${location.host}/ws`);

setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN) return;

    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 640; tmpCanvas.height = 480;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(video, 0, 0, 640, 480);

    const quality = parseFloat(qualitySlider.value);
    const landmarks = drawLandmarks.checked;
    ws.send(JSON.stringify({ 
        image: tmpCanvas.toDataURL('image/jpeg', quality),
        draw_landmarks: landmarks
    }));

}, 60); // ~15 FPS

// Receive response from server (JSON with image and labels)
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    // 1. Update the canvas with the processed image
    const img = new Image();
    img.onload = () => ctx.drawImage(img, 0, 0);
    img.src = data.image;

    // 2. Update labels and FPS on frontend
    const labelsDiv = document.getElementById('labels');
    const fpsValue = document.getElementById('fps-value');
    
    if (fpsValue && data.fps !== undefined) {
        fpsValue.innerText = data.fps;
    }

    if (data.labels && data.labels.length > 0) {
        labelsDiv.innerHTML = data.labels.map(res =>
            `<span class="label-item">
                ${res.label}: ${res.gesture} (${(res.confidence * 100).toFixed(1)}%)
             </span>`
        ).join('');
    } else {
        labelsDiv.innerHTML = "No hand detected";
    }

    // 3. Match display logic
    const matchContainer = document.getElementById('match-container');
    const matchImage = document.getElementById('match-image');

    if (data.match_gesture) {
        // Normalize filename (e.g., 'Rock' -> 'rock.png')
        const fileName = data.match_gesture.toLowerCase().replace(/ /g, '_') + '.png';
        matchImage.src = `/assets/images/gestures/${fileName}`;
        matchContainer.classList.add('visible');
    } else {
        matchContainer.classList.remove('visible');
        // Do not clear src immediately to avoid flickers during fadeout
    }
};
