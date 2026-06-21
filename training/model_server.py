from __future__ import annotations

import base64
import binascii
import csv
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template_string, request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "violence_mobilenet_gru.keras"
INCIDENT_LOG_PATH = ROOT / "logs" / "incidents.csv"
THRESHOLD = 0.70
LIVE_THRESHOLD = 0.70
LIVE_CONFIRMATION_WINDOW = 1
LIVE_CONFIRMATION_HITS = 1
INCIDENT_LOG_COOLDOWN_SECONDS = 30.0
WINDOWS = 12
WINDOW_SECONDS = 3.0

app = Flask(__name__)
model = tf.keras.models.load_model(str(MODEL_PATH))
_, SEQUENCE_LENGTH, IMAGE_SIZE, _, _ = model.input_shape
last_incident_log_monotonic = 0.0


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Violence Detection Model</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --surface: #fff;
      --line: #d8e0e8;
      --text: #17202a;
      --muted: #66717f;
      --blue: #1f6feb;
      --red: #c5221f;
      --green: #15803d;
      --dark: #111827;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .login-view {
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .login-panel {
      width: min(100%, 420px);
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 50px rgba(23,32,42,.12);
      padding: 24px;
    }
    .login-panel h1 {
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }
    .login-panel p {
      margin: 0 0 22px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      font-weight: 650;
    }
    .field {
      display: grid;
      gap: 7px;
      margin-bottom: 14px;
      color: var(--text);
      font-size: 13px;
      font-weight: 750;
    }
    .field input {
      width: 100%;
      min-height: 44px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 12px;
      color: var(--text);
      background: #fbfcfe;
      font: inherit;
      font-weight: 650;
    }
    .field input:focus {
      border-color: var(--blue);
      outline: 3px solid rgba(31,111,235,.14);
    }
    .field input:invalid:not(:placeholder-shown) {
      border-color: var(--red);
    }
    .login-error {
      min-height: 20px;
      margin: 4px 0 14px;
      color: var(--red);
      font-size: 13px;
      font-weight: 700;
    }
    .app-root {
      min-height: 100vh;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 24px clamp(18px, 4vw, 44px);
      background: rgba(255,255,255,.9);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
    .user-pill {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 750;
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 22px;
      padding: 24px clamp(18px, 4vw, 44px);
    }
    section, aside {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 50px rgba(23,32,42,.12);
    }
    .viewer {
      position: relative;
      height: clamp(300px, 62vh, 680px);
      background: var(--dark);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      border-radius: 8px 8px 0 0;
      border: 0 solid transparent;
      transition: box-shadow .18s ease, border-color .18s ease;
    }
    .viewer.alert-active {
      border: 8px solid var(--red);
      box-shadow: 0 0 0 4px rgba(197, 34, 31, .22), 0 18px 50px rgba(197, 34, 31, .32);
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      object-position: center;
      background: var(--dark);
    }
    .live-alert-overlay {
      position: absolute;
      top: 18px;
      left: 18px;
      right: 18px;
      padding: 18px 20px;
      border-radius: 8px;
      background: rgba(197, 34, 31, .93);
      color: #fff;
      box-shadow: 0 14px 38px rgba(0,0,0,.32);
      z-index: 2;
      pointer-events: none;
    }
    .live-alert-overlay strong {
      display: block;
      font-size: clamp(24px, 4vw, 42px);
      line-height: 1.05;
      letter-spacing: 0;
    }
    .live-alert-overlay span {
      display: block;
      margin-top: 8px;
      font-size: clamp(18px, 2vw, 26px);
      font-weight: 800;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--line);
      background: #fbfcfe;
    }
    .media-insights {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(260px, 360px);
      gap: 14px;
      padding: 14px;
      border-top: 1px solid var(--line);
      background: #fbfcfe;
      border-radius: 0 0 8px 8px;
    }
    .insight-panel {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .insight-panel span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      font-weight: 750;
    }
    .insight-panel strong {
      display: block;
      margin-top: 6px;
      font-size: 24px;
      line-height: 1;
    }
    button, input::file-selector-button {
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      cursor: pointer;
      padding: 0 14px;
      font-weight: 700;
    }
    button.primary {
      border-color: var(--blue);
      background: var(--blue);
      color: #fff;
    }
    button:disabled { opacity: .5; cursor: wait; }
    aside { padding: 18px; }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 92px;
      background: #fbfcfe;
      margin-bottom: 12px;
    }
    .metric span {
      color: var(--muted);
      font-size: 13px;
      font-weight: 750;
    }
    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: 28px;
      line-height: 1;
    }
    .violence strong { color: var(--red); }
    .safe strong { color: var(--green); }
    progress {
      width: 100%;
      height: 12px;
      accent-color: var(--blue);
    }
    .graph-canvas {
      display: block;
      width: 100%;
      height: 132px;
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .evidence-frame {
      width: 100%;
      aspect-ratio: 16 / 9;
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      object-fit: cover;
      background: #eef2f7;
    }
    .evidence-frame.empty {
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 13px;
      font-weight: 750;
    }
    .muted {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      font-weight: 650;
    }
    @media (max-width: 900px) {
      header { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; }
      .viewer { height: clamp(260px, 54vh, 560px); }
      .user-pill { white-space: normal; }
      .media-insights { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <section id="loginView" class="login-view">
    <form id="loginForm" class="login-panel" novalidate>
      <h1>Sign in</h1>
      <p>Enter your name and a valid email to continue.</p>
      <label class="field" for="nameInput">
        Name
        <input id="nameInput" name="name" type="text" autocomplete="name" minlength="2" placeholder="Your name" required>
      </label>
      <label class="field" for="emailInput">
        Email
        <input id="emailInput" name="email" type="email" autocomplete="email" inputmode="email" placeholder="name@example.com" pattern="^[^\\s@]+@[^\\s@]+\\.[^\\s@]{2,}$" required>
      </label>
      <div id="loginError" class="login-error" aria-live="polite"></div>
      <button class="primary" type="submit">Continue</button>
    </form>
  </section>

  <div id="appRoot" class="app-root" hidden>
    <header>
      <h1>Violence Detection Model</h1>
      <div class="user-pill">
        <span id="signedInUser"></span>
        <button id="logout" type="button">Logout</button>
      </div>
    </header>
    <main>
      <section>
        <div id="viewer" class="viewer">
          <video id="preview" controls playsinline></video>
          <div id="liveAlertOverlay" class="live-alert-overlay" hidden>
            <strong>🚨 VIOLENCE DETECTED 🚨</strong>
            <span id="liveAlertConfidence">Confidence: 0%</span>
          </div>
        </div>
        <div class="controls">
          <input id="file" type="file" accept="video/*">
          <button id="play" type="button">Play</button>
          <button id="pause" type="button">Pause</button>
          <button id="predict" class="primary" type="button">Predict</button>
          <button id="camera" type="button">Camera</button>
          <button id="stopLive" type="button" disabled>Stop live</button>
        </div>
        <div class="media-insights">
          <div class="insight-panel">
            <span>Live peak graph</span>
            <strong id="livePeak">0.000</strong>
            <canvas id="liveGraph" class="graph-canvas" width="640" height="132"></canvas>
          </div>
          <div class="insight-panel">
            <span>High-risk frame</span>
            <strong id="evidenceScore">No frame</strong>
            <div id="evidenceEmpty" class="evidence-frame empty">Waiting for high-risk frame</div>
            <img id="evidenceFrame" class="evidence-frame" alt="Highest-risk frame" hidden>
            <p class="muted" id="evidenceDetail">The highest-risk frame appears here for uploads and live detection.</p>
          </div>
        </div>
      </section>
      <aside>
        <div id="predictionBox" class="metric">
          <span>Prediction</span>
          <strong id="label">No video</strong>
        </div>
        <div class="metric">
          <span>Violence probability</span>
          <strong id="probability">0.000</strong>
          <progress id="bar" max="1" value="0"></progress>
        </div>
        <div class="metric">
          <span>Window scan</span>
          <strong id="windows">12 windows</strong>
          <p class="muted">The model scans multiple 3-second windows and uses the highest violence probability.</p>
        </div>
        <div class="metric">
          <span>Live stream</span>
          <strong id="liveState">Idle</strong>
          <p class="muted" id="liveDetail">Start the camera to run rolling model predictions.</p>
        </div>
        <p class="muted" id="status">Model threshold: 0.70</p>
      </aside>
    </main>
  </div>
  <script>
    const loginView = document.getElementById("loginView");
    const appRoot = document.getElementById("appRoot");
    const loginForm = document.getElementById("loginForm");
    const nameInput = document.getElementById("nameInput");
    const emailInput = document.getElementById("emailInput");
    const loginError = document.getElementById("loginError");
    const signedInUser = document.getElementById("signedInUser");
    const logoutBtn = document.getElementById("logout");
    const fileInput = document.getElementById("file");
    const viewer = document.getElementById("viewer");
    const preview = document.getElementById("preview");
    const liveAlertOverlay = document.getElementById("liveAlertOverlay");
    const liveAlertConfidence = document.getElementById("liveAlertConfidence");
    const playBtn = document.getElementById("play");
    const pauseBtn = document.getElementById("pause");
    const predictBtn = document.getElementById("predict");
    const cameraBtn = document.getElementById("camera");
    const stopLiveBtn = document.getElementById("stopLive");
    const label = document.getElementById("label");
    const probability = document.getElementById("probability");
    const bar = document.getElementById("bar");
    const windowsEl = document.getElementById("windows");
    const liveState = document.getElementById("liveState");
    const liveDetail = document.getElementById("liveDetail");
    const livePeak = document.getElementById("livePeak");
    const liveGraph = document.getElementById("liveGraph");
    const evidenceScore = document.getElementById("evidenceScore");
    const evidenceEmpty = document.getElementById("evidenceEmpty");
    const evidenceFrame = document.getElementById("evidenceFrame");
    const evidenceDetail = document.getElementById("evidenceDetail");
    const statusEl = document.getElementById("status");
    const predictionBox = document.getElementById("predictionBox");
    const sequenceLength = {{ sequence_length }};
    const imageSize = {{ image_size }};
    const modelThreshold = {{ threshold }};
    const liveThreshold = {{ live_threshold }};
    const liveConfirmationWindow = {{ live_confirmation_window }};
    const liveConfirmationHits = {{ live_confirmation_hits }};
    const liveCanvas = document.createElement("canvas");
    const liveCtx = liveCanvas.getContext("2d", { willReadFrequently: true });
    const evidenceCanvas = document.createElement("canvas");
    const evidenceCtx = evidenceCanvas.getContext("2d");
    const graphCtx = liveGraph.getContext("2d");
    let currentUrl = "";
    let liveStream = null;
    let liveFrameTimer = 0;
    let livePredictTimer = 0;
    let liveBusy = false;
    let liveFrames = [];
    let liveProbabilities = [];
    let liveSamples = [];
    let livePeakProbability = 0;
    let liveStartedAt = 0;
    let latestEvidenceFrame = "";
    let activeThreshold = modelThreshold;
    let bestConfirmedCandidate = { probability: 0, src: "", time: 0 };
    let alertActive = false;
    let alarmContext = null;
    let alarmOscillator = null;
    let alarmGain = null;
    let incidentLogBusy = false;
    const emailPattern = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]{2,}$/i;

    liveCanvas.width = imageSize;
    liveCanvas.height = imageSize;
    evidenceCanvas.width = 320;
    evidenceCanvas.height = 180;

    function isValidEmail(email) {
      return emailPattern.test(email.trim());
    }

    function showApp(user) {
      signedInUser.textContent = `${user.name} (${user.email})`;
      loginView.hidden = true;
      appRoot.hidden = false;
    }

    function showLogin() {
      appRoot.hidden = true;
      loginView.hidden = false;
      loginForm.reset();
      loginError.textContent = "";
      nameInput.focus();
    }

    async function playAlarm() {
      if (alarmOscillator) {
        return;
      }

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) {
        return;
      }

      alarmContext = alarmContext || new AudioContextClass();
      if (alarmContext.state === "suspended") {
        await alarmContext.resume();
      }

      alarmOscillator = alarmContext.createOscillator();
      alarmGain = alarmContext.createGain();
      alarmOscillator.type = "square";
      alarmOscillator.frequency.value = 880;
      alarmGain.gain.value = 0.08;
      alarmOscillator.connect(alarmGain);
      alarmGain.connect(alarmContext.destination);
      alarmOscillator.start();
    }

    function stopAlarm() {
      if (!alarmOscillator) {
        return;
      }

      alarmOscillator.stop();
      alarmOscillator.disconnect();
      if (alarmGain) {
        alarmGain.disconnect();
      }
      alarmOscillator = null;
      alarmGain = null;
    }

    async function logIncident(confidence) {
      if (incidentLogBusy) {
        return;
      }

      incidentLogBusy = true;
      try {
        await fetch("/log_incident", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confidence })
        });
      } catch (error) {
        console.warn("Could not log incident", error);
      } finally {
        incidentLogBusy = false;
      }
    }

    function setLiveAlert(active, confidence) {
      alertActive = active;
      viewer.classList.toggle("alert-active", active);
      liveAlertOverlay.hidden = !active;
      liveAlertConfidence.textContent = `Confidence: ${Math.round(confidence * 100)}%`;

      if (active) {
        playAlarm().catch((error) => console.warn("Could not start alarm", error));
      } else {
        stopAlarm();
      }
    }

    function formatLiveTime(seconds) {
      const safeSeconds = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
      const minutes = Math.floor(safeSeconds / 60);
      const secs = Math.floor(safeSeconds % 60);
      return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    }

    function drawLiveGraph() {
      const width = liveGraph.width;
      const height = liveGraph.height;
      const graphTop = 14;
      const graphBottom = height - 24;
      const graphHeight = graphBottom - graphTop;
      const thresholdY = graphBottom - (activeThreshold * graphHeight);
      const samples = liveSamples.slice(-45);

      graphCtx.clearRect(0, 0, width, height);
      graphCtx.fillStyle = "#ffffff";
      graphCtx.fillRect(0, 0, width, height);

      graphCtx.strokeStyle = "#e1e7ef";
      graphCtx.lineWidth = 1;
      for (let index = 0; index <= 4; index += 1) {
        const y = graphTop + (graphHeight * index) / 4;
        graphCtx.beginPath();
        graphCtx.moveTo(0, y);
        graphCtx.lineTo(width, y);
        graphCtx.stroke();
      }

      graphCtx.strokeStyle = "#c5221f";
      graphCtx.setLineDash([5, 5]);
      graphCtx.beginPath();
      graphCtx.moveTo(0, thresholdY);
      graphCtx.lineTo(width, thresholdY);
      graphCtx.stroke();
      graphCtx.setLineDash([]);

      graphCtx.fillStyle = "#66717f";
      graphCtx.font = "11px system-ui, sans-serif";
      graphCtx.fillText(`Threshold ${activeThreshold.toFixed(2)}`, 8, Math.max(12, thresholdY - 6));

      if (!samples.length) {
        graphCtx.fillStyle = "#66717f";
        graphCtx.font = "13px system-ui, sans-serif";
        graphCtx.fillText("Upload or start camera to see probabilities", 14, height / 2);
        return;
      }

      graphCtx.strokeStyle = "#1f6feb";
      graphCtx.lineWidth = 3;
      graphCtx.beginPath();
      samples.forEach((sample, index) => {
        const x = samples.length === 1 ? width - 12 : (index / (samples.length - 1)) * width;
        const y = graphBottom - (sample.probability * graphHeight);
        if (index === 0) {
          graphCtx.moveTo(x, y);
        } else {
          graphCtx.lineTo(x, y);
        }
      });
      graphCtx.stroke();

      const peak = samples.reduce((best, sample, index) => {
        return sample.probability > best.sample.probability ? { sample, index } : best;
      }, { sample: samples[0], index: 0 });
      const peakX = samples.length === 1 ? width - 12 : (peak.index / (samples.length - 1)) * width;
      const peakY = graphBottom - (peak.sample.probability * graphHeight);
      graphCtx.fillStyle = "#c5221f";
      graphCtx.beginPath();
      graphCtx.arc(peakX, peakY, 5, 0, Math.PI * 2);
      graphCtx.fill();

      graphCtx.fillStyle = "#17202a";
      graphCtx.font = "12px system-ui, sans-serif";
      graphCtx.fillText(`Peak ${peak.sample.probability.toFixed(3)}`, 8, height - 7);
    }

    function resetLiveTracking() {
      liveSamples = [];
      livePeakProbability = 0;
      latestEvidenceFrame = "";
      bestConfirmedCandidate = { probability: 0, src: "", time: 0 };
      livePeak.textContent = "0.000";
      evidenceScore.textContent = "No frame";
      evidenceFrame.hidden = true;
      evidenceFrame.removeAttribute("src");
      evidenceEmpty.hidden = false;
      evidenceDetail.textContent = "The highest-risk frame appears here for uploads and live detection.";
      drawLiveGraph();
    }

    function updateEvidenceFrame(probabilityValue, frameSrc, seconds, sourceLabel, requireThreshold = true) {
      if (!frameSrc || (requireThreshold && probabilityValue < activeThreshold)) {
        return;
      }

      evidenceFrame.src = frameSrc;
      evidenceFrame.hidden = false;
      evidenceEmpty.hidden = true;
      evidenceScore.textContent = probabilityValue.toFixed(4);
      const thresholdText = probabilityValue >= activeThreshold ? "above threshold" : "highest scanned window";
      evidenceDetail.textContent = `${sourceLabel} at ${formatLiveTime(seconds)} (${thresholdText}).`;
    }

    function resetPredictionDisplay() {
      label.textContent = "Ready";
      probability.textContent = "0.000";
      bar.value = 0;
      predictionBox.className = "metric";
      windowsEl.textContent = "12 windows";
      statusEl.textContent = `Model threshold: ${modelThreshold.toFixed(2)}`;
    }

    function stopLive(message = "Live stopped") {
      if (liveFrameTimer) {
        window.clearInterval(liveFrameTimer);
        liveFrameTimer = 0;
      }
      if (livePredictTimer) {
        window.clearInterval(livePredictTimer);
        livePredictTimer = 0;
      }
      if (liveStream) {
        liveStream.getTracks().forEach((track) => track.stop());
        liveStream = null;
      }
      liveBusy = false;
      liveFrames = [];
      liveProbabilities = [];
      liveStartedAt = 0;
      setLiveAlert(false, 0);
      preview.srcObject = null;
      preview.controls = true;
      cameraBtn.disabled = false;
      stopLiveBtn.disabled = true;
      predictBtn.disabled = false;
      liveState.textContent = "Idle";
      liveDetail.textContent = message;
    }

    function captureLiveFrame() {
      if (!liveStream || preview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        return;
      }

      liveCtx.drawImage(preview, 0, 0, imageSize, imageSize);
      liveFrames.push(liveCanvas.toDataURL("image/jpeg", 0.72));
      liveFrames = liveFrames.slice(-sequenceLength);
      evidenceCtx.drawImage(preview, 0, 0, evidenceCanvas.width, evidenceCanvas.height);
      latestEvidenceFrame = evidenceCanvas.toDataURL("image/jpeg", 0.78);

      if (liveFrames.length < sequenceLength) {
        liveState.textContent = "Buffering";
        liveDetail.textContent = `Collected ${liveFrames.length}/${sequenceLength} frames.`;
      } else if (!liveBusy) {
        liveState.textContent = "Scanning";
        liveDetail.textContent = `Running ${sequenceLength}-frame rolling predictions.`;
      }
    }

    async function predictLiveWindow() {
      if (liveBusy || liveFrames.length < sequenceLength) {
        return;
      }

      liveBusy = true;
      const evidenceAtRequest = latestEvidenceFrame;
      const secondsAtRequest = liveStartedAt ? (performance.now() - liveStartedAt) / 1000 : 0;
      try {
        const response = await fetch("/predict_frames", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frames: liveFrames })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Live prediction failed");

        const probabilityValue = data.probability;
        liveProbabilities.push(probabilityValue);
        liveProbabilities = liveProbabilities.slice(-liveConfirmationWindow);
        liveSamples.push({ probability: probabilityValue, time: secondsAtRequest });
        liveSamples = liveSamples.slice(-120);
        const recent = liveProbabilities.slice(-liveConfirmationWindow);
        const alertHits = recent.filter((item) => item >= data.threshold).length;
        const confirmed = alertHits >= liveConfirmationHits;
        const isNewPeak = probabilityValue >= livePeakProbability;
        livePeakProbability = Math.max(livePeakProbability, probabilityValue);
        livePeak.textContent = livePeakProbability.toFixed(4);
        if (probabilityValue >= data.threshold && probabilityValue >= bestConfirmedCandidate.probability) {
          bestConfirmedCandidate = {
            probability: probabilityValue,
            src: evidenceAtRequest,
            time: secondsAtRequest
          };
        }
        if (confirmed) {
          updateEvidenceFrame(
            bestConfirmedCandidate.probability,
            bestConfirmedCandidate.src,
            bestConfirmedCandidate.time,
            "Confirmed live frame",
            false
          );
          setLiveAlert(true, probabilityValue);
          logIncident(probabilityValue);
        } else {
          setLiveAlert(false, probabilityValue);
        }
        drawLiveGraph();

        label.textContent = confirmed ? "VIOLENCE" : "NON_VIOLENCE";
        probability.textContent = probabilityValue.toFixed(4);
        bar.value = probabilityValue;
        windowsEl.textContent = `${data.frames_used} live frames`;
        predictionBox.className = confirmed ? "metric violence" : "metric safe";
        liveState.textContent = confirmed ? "Alert" : "Scanning";
        liveDetail.textContent = `Current ${probabilityValue.toFixed(4)} | threshold ${data.threshold}`;
        statusEl.textContent = confirmed
          ? "Violence detected in live video."
          : "Live camera prediction running.";
      } catch (error) {
        liveState.textContent = "Error";
        liveDetail.textContent = error.message;
      } finally {
        liveBusy = false;
      }
    }

    async function startLiveCamera() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        liveState.textContent = "Unavailable";
        liveDetail.textContent = "This browser cannot access a camera.";
        return;
      }

      stopLive("Starting live camera...");
      if (currentUrl) {
        URL.revokeObjectURL(currentUrl);
        currentUrl = "";
      }
      preview.removeAttribute("src");
      preview.load();
      fileInput.value = "";
      predictBtn.disabled = true;
      resetPredictionDisplay();
      activeThreshold = liveThreshold;
      resetLiveTracking();
      liveState.textContent = "Starting";
      liveDetail.textContent = "Waiting for camera permission.";

      try {
        liveStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 960 }, height: { ideal: 540 } },
          audio: false
        });
        preview.srcObject = liveStream;
        preview.controls = false;
        await preview.play();
        liveStartedAt = performance.now();
        cameraBtn.disabled = true;
        stopLiveBtn.disabled = false;
        label.textContent = "LIVE";
        liveState.textContent = "Buffering";
        liveDetail.textContent = `Collecting ${sequenceLength} frames.`;
        statusEl.textContent = `Live camera running. Alert threshold: ${liveThreshold.toFixed(2)}.`;
        captureLiveFrame();
        liveFrameTimer = window.setInterval(captureLiveFrame, 300);
        livePredictTimer = window.setInterval(predictLiveWindow, 900);
      } catch (error) {
        stopLive("Camera blocked or unavailable.");
        liveState.textContent = "Blocked";
        liveDetail.textContent = error.message || "Camera blocked or unavailable.";
      }
    }

    let savedUser = null;
    try {
      savedUser = JSON.parse(localStorage.getItem("violenceDemoUser") || "null");
    } catch (error) {
      localStorage.removeItem("violenceDemoUser");
    }
    if (savedUser && savedUser.name && isValidEmail(savedUser.email || "")) {
      showApp(savedUser);
    } else {
      showLogin();
    }

    loginForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const name = nameInput.value.trim();
      const email = emailInput.value.trim();

      if (name.length < 2) {
        loginError.textContent = "Enter your name.";
        nameInput.focus();
        return;
      }

      if (!isValidEmail(email)) {
        loginError.textContent = "Enter a valid email address.";
        emailInput.focus();
        return;
      }

      const user = { name, email };
      localStorage.setItem("violenceDemoUser", JSON.stringify(user));
      showApp(user);
    });

    logoutBtn.addEventListener("click", () => {
      localStorage.removeItem("violenceDemoUser");
      stopLive("Signed out.");
      activeThreshold = modelThreshold;
      resetLiveTracking();
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      currentUrl = "";
      preview.removeAttribute("src");
      preview.load();
      showLogin();
    });

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      stopLive("Upload mode ready.");
      activeThreshold = modelThreshold;
      resetLiveTracking();
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      currentUrl = URL.createObjectURL(file);
      preview.src = currentUrl;
      resetPredictionDisplay();
    });

    playBtn.addEventListener("click", () => preview.play());
    pauseBtn.addEventListener("click", () => preview.pause());
    cameraBtn.addEventListener("click", startLiveCamera);
    stopLiveBtn.addEventListener("click", () => stopLive("Live stopped."));

    predictBtn.addEventListener("click", async () => {
      const file = fileInput.files[0];
      if (!file) return;
      const body = new FormData();
      body.append("video", file);
      predictBtn.disabled = true;
      label.textContent = "Analyzing";
      statusEl.textContent = "Scanning video windows...";
      try {
        const response = await fetch("/predict", { method: "POST", body });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed");
        label.textContent = data.label;
        probability.textContent = data.max_probability.toFixed(4);
        bar.value = data.max_probability;
        windowsEl.textContent = `${data.windows} windows`;
        predictionBox.className = data.label === "VIOLENCE" ? "metric violence" : "metric safe";
        statusEl.textContent = `Mean probability: ${data.mean_probability.toFixed(4)} | threshold: ${data.threshold}`;
        activeThreshold = data.threshold;
        liveSamples = (data.timeline || []).map((item) => ({
          probability: item.probability,
          time: item.start_seconds
        }));
        livePeakProbability = data.max_probability;
        livePeak.textContent = livePeakProbability.toFixed(4);
        drawLiveGraph();
        updateEvidenceFrame(
          data.max_probability,
          data.evidence_frame,
          data.peak_time || 0,
          "Uploaded video frame",
          false
        );
      } catch (error) {
        label.textContent = "Error";
        statusEl.textContent = error.message;
      } finally {
        predictBtn.disabled = false;
      }
    });
    resetLiveTracking();
  </script>
</body>
</html>
"""


def read_video_window(
    capture: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
) -> np.ndarray:
    positions = np.linspace(start_frame, max(start_frame, end_frame), SEQUENCE_LENGTH).astype(int)
    frames: list[np.ndarray] = []
    fallback = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if not ok or frame is None:
            frame_rgb = fallback.copy()
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            fallback = frame_rgb
        frames.append(frame_rgb)

    return preprocess_input(np.asarray(frames, dtype=np.float32))


def encode_evidence_frame(capture: cv2.VideoCapture, frame_index: int) -> str:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_index)))
    ok, frame = capture.read()
    if not ok or frame is None:
        return ""

    frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ok:
        return ""
    payload = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def predict_file(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        safe_total = max(1, total_frames)
        window_frames = int(max(SEQUENCE_LENGTH, round(fps * WINDOW_SECONDS)))

        if safe_total <= window_frames:
            starts = [0]
        else:
            max_start = max(0, safe_total - window_frames)
            starts = np.linspace(0, max_start, WINDOWS).astype(int).tolist()

        clips = np.asarray(
            [
                read_video_window(
                    capture,
                    start_frame=start,
                    end_frame=min(safe_total - 1, start + window_frames),
                )
                for start in starts
            ],
            dtype=np.float32,
        )

        probabilities = model.predict(clips, verbose=0).reshape(-1)
        peak_index = int(np.argmax(probabilities))
        peak_start = int(starts[peak_index])
        peak_frame = min(safe_total - 1, peak_start + window_frames // 2)
        peak_time = peak_frame / fps
        max_probability = float(np.max(probabilities))
        mean_probability = float(np.mean(probabilities))
        label = "VIOLENCE" if max_probability >= THRESHOLD else "NON_VIOLENCE"
        timeline = [
            {
                "start_seconds": float(start / fps),
                "end_seconds": float(min(safe_total - 1, start + window_frames) / fps),
                "probability": float(probability),
            }
            for start, probability in zip(starts, probabilities)
        ]

        return {
            "label": label,
            "max_probability": max_probability,
            "mean_probability": mean_probability,
            "threshold": THRESHOLD,
            "windows": int(len(probabilities)),
            "timeline": timeline,
            "peak_time": float(peak_time),
            "evidence_frame": encode_evidence_frame(capture, peak_frame),
        }
    finally:
        capture.release()


def decode_frame_data(frame_data: str) -> np.ndarray:
    if not isinstance(frame_data, str) or not frame_data:
        raise ValueError("Each live frame must be a non-empty image string.")

    payload = frame_data.split(",", 1)[1] if "," in frame_data else frame_data
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid live frame image data.") from error

    buffer = np.frombuffer(raw, dtype=np.uint8)
    frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Could not decode a live frame image.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def predict_frame_sequence(encoded_frames: list[str]) -> dict[str, float | int | str]:
    if not isinstance(encoded_frames, list):
        raise ValueError("Expected a JSON array of live frames.")
    if len(encoded_frames) < SEQUENCE_LENGTH:
        raise ValueError(f"Need at least {SEQUENCE_LENGTH} live frames.")

    positions = np.linspace(0, len(encoded_frames) - 1, SEQUENCE_LENGTH).astype(int)
    frames = [decode_frame_data(encoded_frames[int(position)]) for position in positions]
    clip = preprocess_input(np.asarray(frames, dtype=np.float32))[None, ...]
    probability = float(model.predict(clip, verbose=0).reshape(-1)[0])
    label = "VIOLENCE" if probability >= LIVE_THRESHOLD else "NON_VIOLENCE"

    return {
        "label": label,
        "probability": probability,
        "threshold": LIVE_THRESHOLD,
        "frames_used": SEQUENCE_LENGTH,
        "confirmation_window": LIVE_CONFIRMATION_WINDOW,
        "confirmation_hits": LIVE_CONFIRMATION_HITS,
    }


def ensure_incident_log() -> None:
    INCIDENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not INCIDENT_LOG_PATH.exists():
        with INCIDENT_LOG_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Confidence"])


def log_incident(confidence: float) -> bool:
    """Append one live incident row, throttled to avoid duplicate continuous alerts."""
    global last_incident_log_monotonic

    now_monotonic = time.monotonic()
    if (
        last_incident_log_monotonic
        and now_monotonic - last_incident_log_monotonic < INCIDENT_LOG_COOLDOWN_SECONDS
    ):
        return False

    ensure_incident_log()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with INCIDENT_LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, f"{confidence:.4f}"])

    last_incident_log_monotonic = now_monotonic
    return True


@app.get("/")
def index() -> str:
    return render_template_string(
        PAGE,
        image_size=IMAGE_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        threshold=THRESHOLD,
        live_threshold=LIVE_THRESHOLD,
        live_confirmation_window=LIVE_CONFIRMATION_WINDOW,
        live_confirmation_hits=LIVE_CONFIRMATION_HITS,
    )


@app.post("/predict")
def predict() -> tuple[object, int] | object:
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "Upload a video file first."}), 400

    suffix = Path(secure_filename(upload.filename)).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        upload.save(temp_file)

    try:
        return jsonify(predict_file(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/predict_frames")
def predict_frames() -> tuple[object, int] | object:
    payload = request.get_json(silent=True) or {}
    frames = payload.get("frames")

    try:
        return jsonify(predict_frame_sequence(frames))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/log_incident")
def log_live_incident() -> tuple[object, int] | object:
    payload = request.get_json(silent=True) or {}

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return jsonify({"error": "A numeric confidence is required."}), 400

    if confidence < LIVE_THRESHOLD:
        return jsonify({"logged": False, "reason": "below_threshold", "threshold": LIVE_THRESHOLD})

    logged = log_incident(confidence)
    return jsonify({
        "logged": logged,
        "cooldown_seconds": INCIDENT_LOG_COOLDOWN_SECONDS,
        "path": str(INCIDENT_LOG_PATH.relative_to(ROOT)),
    })


if __name__ == "__main__":
    ensure_incident_log()
    app.run(host="127.0.0.1", port=5001, debug=False)
