(function () {
  "use strict";

  const LIVE_INTERVAL_MS = 250;
  const ANALYSIS_WIDTH = 192;
  const ANALYSIS_HEIGHT = 108;
  const MAX_VIDEO_SECONDS = 180;
  const ANALYSIS_PROFILES = {
    fast: { label: "Fast", fps: 1.5, maxFrames: 45, seekTimeout: 350 },
    balanced: { label: "Balanced", fps: 3, maxFrames: 120, seekTimeout: 650 },
    accurate: { label: "Accurate", fps: 5, maxFrames: 240, seekTimeout: 900 }
  };
  const SENSITIVITY_PROFILES = {
    normal: { label: "Normal", multiplier: 1.0, boost: 0, thresholdOffset: 0 },
    high: { label: "High", multiplier: 1.22, boost: 6, thresholdOffset: -6 },
    cctv: { label: "CCTV", multiplier: 1.42, boost: 10, thresholdOffset: -12 }
  };
  const TILE_COLS = 12;
  const TILE_ROWS = 8;
  const TRAINING_STORAGE_KEY = "violence-demo-training-samples";
  const MODEL_STORAGE_KEY = "violence-demo-custom-model";
  const USER_STORAGE_KEY = "violence-demo-user";

  const els = {
    loginView: document.getElementById("loginView"),
    loginForm: document.getElementById("loginForm"),
    nameInput: document.getElementById("nameInput"),
    emailInput: document.getElementById("emailInput"),
    loginError: document.getElementById("loginError"),
    signedInUser: document.getElementById("signedInUser"),
    logoutBtn: document.getElementById("logoutBtn"),
    video: document.getElementById("videoPreview"),
    videoInput: document.getElementById("videoInput"),
    uploadBtn: document.getElementById("uploadBtn"),
    analyzeBtn: document.getElementById("analyzeBtn"),
    playBtn: document.getElementById("playBtn"),
    pauseBtn: document.getElementById("pauseBtn"),
    cameraBtn: document.getElementById("cameraBtn"),
    stopBtn: document.getElementById("stopBtn"),
    calmSampleBtn: document.getElementById("calmSampleBtn"),
    riskSampleBtn: document.getElementById("riskSampleBtn"),
    exportBtn: document.getElementById("exportBtn"),
    emptyState: document.getElementById("emptyState"),
    alertBanner: document.getElementById("alertBanner"),
    statusText: document.getElementById("statusText"),
    predictionText: document.getElementById("predictionText"),
    predictionDetail: document.getElementById("predictionDetail"),
    currentRisk: document.getElementById("currentRisk"),
    peakRisk: document.getElementById("peakRisk"),
    averageRisk: document.getElementById("averageRisk"),
    frameCount: document.getElementById("frameCount"),
    thresholdInput: document.getElementById("thresholdInput"),
    thresholdValue: document.getElementById("thresholdValue"),
    analysisDepth: document.getElementById("analysisDepth"),
    sensitivityMode: document.getElementById("sensitivityMode"),
    markNonViolenceBtn: document.getElementById("markNonViolenceBtn"),
    markViolenceBtn: document.getElementById("markViolenceBtn"),
    trainCustomBtn: document.getElementById("trainCustomBtn"),
    clearTrainingBtn: document.getElementById("clearTrainingBtn"),
    trainingStatus: document.getElementById("trainingStatus"),
    motionBar: document.getElementById("motionBar"),
    impactBar: document.getElementById("impactBar"),
    instabilityBar: document.getElementById("instabilityBar"),
    durationLabel: document.getElementById("durationLabel"),
    segmentCount: document.getElementById("segmentCount"),
    evidenceGrid: document.getElementById("evidenceGrid"),
    segmentList: document.getElementById("segmentList"),
    timeline: document.getElementById("timelineCanvas"),
    analysisCanvas: document.getElementById("analysisCanvas"),
    demoPreview: document.getElementById("demoPreview")
  };

  const analysisCtx = els.analysisCanvas.getContext("2d", { willReadFrequently: true });
  const timelineCtx = els.timeline.getContext("2d");
  const demoCtx = els.demoPreview.getContext("2d");

  const state = {
    threshold: Number(els.thresholdInput.value),
    analysisMode: els.analysisDepth.value,
    sensitivityMode: els.sensitivityMode.value,
    scores: [],
    evidence: [],
    segments: [],
    prevLuma: null,
    prevEdge: 0,
    prevBrightness: 0,
    lastRisk: 0,
    loadedName: "",
    loadedDuration: 0,
    videoUrl: "",
    stream: null,
    liveTimer: 0,
    liveStartedAt: 0,
    cancelRequested: false,
    analyzing: false,
    trainingSamples: [],
    customModel: null
  };
  const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/i;

  function isValidEmail(email) {
    return EMAIL_PATTERN.test(email.trim());
  }

  function readSavedUser() {
    try {
      return JSON.parse(localStorage.getItem(USER_STORAGE_KEY) || "null");
    } catch (error) {
      localStorage.removeItem(USER_STORAGE_KEY);
      return null;
    }
  }

  function showAppForUser(user) {
    els.signedInUser.textContent = `${user.name} (${user.email})`;
    els.loginView.hidden = true;
  }

  function showLogin() {
    els.loginView.hidden = false;
    els.loginForm.reset();
    els.loginError.textContent = "";
    els.nameInput.focus();
  }

  function initializeLogin() {
    const savedUser = readSavedUser();
    if (savedUser && savedUser.name && isValidEmail(savedUser.email || "")) {
      showAppForUser(savedUser);
    } else {
      showLogin();
    }
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function formatTime(seconds) {
    if (!Number.isFinite(seconds)) {
      return "00:00";
    }
    const safe = Math.max(0, seconds);
    const minutes = Math.floor(safe / 60);
    const secs = Math.floor(safe % 60);
    return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  function setStatus(message, tone) {
    els.statusText.textContent = message;
    els.statusText.className = `status ${tone || "neutral"}`;
  }

  function setMediaVisible(mode) {
    const showVideo = mode === "video" || mode === "camera";
    const showDemo = mode === "demo";
    els.video.style.display = showVideo ? "block" : "none";
    els.demoPreview.style.display = showDemo ? "block" : "none";
    els.emptyState.classList.toggle("hidden", showVideo || showDemo);
  }

  function syncPlaybackButtons() {
    const canControlVideo = Boolean(state.loadedDuration) && !state.analyzing && !state.stream;
    els.playBtn.disabled = !canControlVideo || !els.video.paused;
    els.pauseBtn.disabled = !canControlVideo || els.video.paused;
  }

  function syncTrainingControls() {
    const hasScores = state.scores.length > 0 && !state.analyzing;
    const positives = state.trainingSamples.filter((sample) => sample.label === 1).length;
    const negatives = state.trainingSamples.filter((sample) => sample.label === 0).length;
    els.markViolenceBtn.disabled = !hasScores;
    els.markNonViolenceBtn.disabled = !hasScores;
    els.trainCustomBtn.disabled = !(positives > 0 && negatives > 0);
    els.clearTrainingBtn.disabled = state.trainingSamples.length === 0 && !state.customModel;

    const modelText = state.customModel ? "Custom model ready" : "No custom model";
    els.trainingStatus.textContent = `${positives} violence, ${negatives} non-violence examples. ${modelText}.`;
  }

  function getSensitivityProfile() {
    return SENSITIVITY_PROFILES[state.sensitivityMode] || SENSITIVITY_PROFILES.normal;
  }

  function getEffectiveThreshold() {
    const profile = getSensitivityProfile();
    return clamp(state.threshold + profile.thresholdOffset, 30, 95);
  }

  function syncThresholdLabel() {
    const effective = getEffectiveThreshold();
    els.thresholdValue.textContent = effective === state.threshold
      ? String(state.threshold)
      : `${state.threshold} / alert ${effective}`;
  }

  function resetSignals() {
    state.scores = [];
    state.evidence = [];
    state.segments = [];
    state.prevLuma = null;
    state.prevEdge = 0;
    state.prevBrightness = 0;
    state.lastRisk = 0;
    state.cancelRequested = false;
    els.alertBanner.classList.add("hidden");
    renderAll();
  }

  function stopCamera() {
    if (state.liveTimer) {
      window.clearInterval(state.liveTimer);
      state.liveTimer = 0;
    }
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    els.video.srcObject = null;
    els.video.controls = true;
    els.stopBtn.disabled = true;
    els.cameraBtn.disabled = false;
    syncPlaybackButtons();
  }

  function stopCurrentWork() {
    state.cancelRequested = true;
    stopCamera();
    state.analyzing = false;
    els.analyzeBtn.disabled = !state.loadedDuration;
    els.stopBtn.disabled = true;
    syncPlaybackButtons();
    syncTrainingControls();
  }

  function scoreFrame(imageData, time) {
    const { data, width, height } = imageData;
    const pixelCount = width * height;
    const luma = new Uint8Array(pixelCount);
    const tileDiffs = new Float32Array(TILE_COLS * TILE_ROWS);
    let diffSum = 0;
    let brightnessSum = 0;
    let redHits = 0;

    for (let i = 0, p = 0, x = 0, yPos = 0; i < data.length; i += 4, p += 1) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const y = (77 * r + 150 * g + 29 * b) >> 8;
      luma[p] = y;
      brightnessSum += y;

      if (state.prevLuma) {
        const diff = Math.abs(y - state.prevLuma[p]);
        const tileX = Math.min(TILE_COLS - 1, Math.floor((x * TILE_COLS) / width));
        const tileY = Math.min(TILE_ROWS - 1, Math.floor((yPos * TILE_ROWS) / height));
        diffSum += diff;
        tileDiffs[tileY * TILE_COLS + tileX] += diff;
      }
      if (r > 95 && r > g * 1.2 && r > b * 1.1) {
        redHits += 1;
      }

      x += 1;
      if (x === width) {
        x = 0;
        yPos += 1;
      }
    }

    let edgeSum = 0;
    for (let y = 1; y < height; y += 1) {
      for (let x = 1; x < width; x += 1) {
        const idx = y * width + x;
        edgeSum += Math.abs(luma[idx] - luma[idx - 1]);
        edgeSum += Math.abs(luma[idx] - luma[idx - width]);
      }
    }

    const motion = state.prevLuma ? diffSum / (255 * pixelCount) : 0;
    const brightness = brightnessSum / pixelCount;
    const brightnessPulse = state.prevLuma ? Math.abs(brightness - state.prevBrightness) / 255 : 0;
    const edge = edgeSum / (255 * pixelCount * 2);
    const edgeJump = state.prevLuma ? Math.abs(edge - state.prevEdge) : 0;
    const redRatio = redHits / pixelCount;
    const tileArea = pixelCount / (TILE_COLS * TILE_ROWS);
    const normalizedTiles = Array.from(tileDiffs, (value) => value / (255 * tileArea));
    normalizedTiles.sort((a, b) => b - a);
    const topTileCount = Math.max(4, Math.ceil(normalizedTiles.length * 0.12));
    const topTileMotion = normalizedTiles
      .slice(0, topTileCount)
      .reduce((total, value) => total + value, 0) / topTileCount;
    const activeTileRatio = normalizedTiles.filter((value) => value > 0.035).length / normalizedTiles.length;

    const globalMotionComponent = clamp((motion - 0.008) / 0.11, 0, 1);
    const localActionComponent = clamp((topTileMotion - 0.018) / 0.135, 0, 1);
    const spreadComponent = clamp(activeTileRatio / 0.22, 0, 1);
    const motionComponent = clamp(
      globalMotionComponent * 0.35 +
      localActionComponent * 0.55 +
      spreadComponent * 0.10,
      0,
      1
    );
    const impactComponent = clamp(
      (edgeJump * 5.4) +
      (brightnessPulse * 2.0) +
      (localActionComponent * 0.20),
      0,
      1
    );
    const instabilityComponent = clamp(
      (redRatio * 2.3) +
      (spreadComponent * 0.45) +
      (localActionComponent * 0.30),
      0,
      1
    );
    const baseRisk = (
      globalMotionComponent * 0.16 +
      localActionComponent * 0.46 +
      impactComponent * 0.24 +
      instabilityComponent * 0.14
    ) * 100;
    const sensitivity = getSensitivityProfile();
    const rawRisk = clamp(baseRisk * sensitivity.multiplier + sensitivity.boost, 0, 100);
    const smoothedRisk = state.prevLuma ? (state.lastRisk * 0.46 + rawRisk * 0.54) : rawRisk;
    const risk = Math.round(clamp(smoothedRisk, 0, 100));

    state.prevLuma = luma;
    state.prevEdge = edge;
    state.prevBrightness = brightness;
    state.lastRisk = risk;

    return {
      time,
      risk,
      motion: Math.round(motionComponent * 100),
      impact: Math.round(impactComponent * 100),
      instability: Math.round(instabilityComponent * 100)
    };
  }

  function captureSource(source) {
    const thumb = document.createElement("canvas");
    thumb.width = 320;
    thumb.height = 180;
    const ctx = thumb.getContext("2d");
    ctx.drawImage(source, 0, 0, thumb.width, thumb.height);
    return thumb.toDataURL("image/jpeg", 0.76);
  }

  function maybeStoreEvidence(result, source) {
    if (result.risk < Math.min(getEffectiveThreshold(), 45)) {
      return;
    }

    const nearbyBetter = state.evidence.some((item) => {
      return Math.abs(item.time - result.time) < 0.75 && item.risk >= result.risk;
    });
    if (nearbyBetter) {
      return;
    }

    state.evidence.push({
      time: result.time,
      risk: result.risk,
      src: captureSource(source)
    });
    state.evidence.sort((a, b) => b.risk - a.risk);
    state.evidence = state.evidence.slice(0, 6);
  }

  function processFrame(source, time) {
    if (source instanceof HTMLVideoElement && source.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }

    analysisCtx.drawImage(source, 0, 0, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
    const imageData = analysisCtx.getImageData(0, 0, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
    const result = scoreFrame(imageData, time);
    state.scores.push(result);
    maybeStoreEvidence(result, source);
    renderAll();
    return result;
  }

  function getStats() {
    if (!state.scores.length) {
      return { current: 0, peak: 0, average: 0 };
    }

    const current = state.scores[state.scores.length - 1].risk;
    const peak = Math.max(...state.scores.map((item) => item.risk));
    const average = Math.round(
      state.scores.reduce((total, item) => total + item.risk, 0) / state.scores.length
    );
    return { current, peak, average };
  }

  function getSampleInterval() {
    if (state.scores.length < 2) {
      return 0.5;
    }

    const gaps = [];
    for (let index = 1; index < state.scores.length; index += 1) {
      const gap = state.scores[index].time - state.scores[index - 1].time;
      if (gap > 0 && Number.isFinite(gap)) {
        gaps.push(gap);
      }
    }

    if (!gaps.length) {
      return 0.5;
    }

    gaps.sort((a, b) => a - b);
    return gaps[Math.floor(gaps.length / 2)];
  }

  function getConfirmationRules() {
    const interval = getSampleInterval();
    const sensitivity = state.sensitivityMode;
    const mode = state.analysisMode;
    const minHits = sensitivity === "cctv" ? 3 : sensitivity === "high" ? 4 : 5;
    const minDuration = mode === "fast" ? 0.8 : sensitivity === "cctv" ? 1.0 : 1.3;

    return {
      interval,
      minHits: mode === "fast" ? Math.max(2, minHits - 2) : minHits,
      minDuration,
      gapLimit: Math.max(0.9, interval * 2.6)
    };
  }

  function computeSegments() {
    const segments = [];
    let active = null;
    const threshold = getEffectiveThreshold();
    const rules = getConfirmationRules();

    state.scores.forEach((item) => {
      if (item.risk >= threshold) {
        if (!active) {
          active = {
            start: item.time,
            end: item.time,
            peak: item.risk,
            hits: 1,
            totalRisk: item.risk,
            totalImpact: item.impact,
            totalInstability: item.instability,
            lastHit: item.time
          };
        } else {
          active.end = item.time;
          active.peak = Math.max(active.peak, item.risk);
          active.hits += 1;
          active.totalRisk += item.risk;
          active.totalImpact += item.impact;
          active.totalInstability += item.instability;
          active.lastHit = item.time;
        }
      } else if (active) {
        const gap = item.time - active.lastHit;
        if (gap > rules.gapLimit) {
          segments.push(active);
          active = null;
        }
      }
    });

    if (active) {
      segments.push(active);
    }

    state.segments = segments
      .map((segment) => ({
        ...segment,
        duration: Math.max(rules.interval, segment.end - segment.start + rules.interval),
        average: Math.round(segment.totalRisk / segment.hits),
        averageImpact: Math.round(segment.totalImpact / segment.hits),
        averageInstability: Math.round(segment.totalInstability / segment.hits)
      }))
      .filter((segment) => {
        const enoughHits = segment.hits >= rules.minHits;
        const enoughDuration = segment.duration >= rules.minDuration;
        const veryStrong = segment.peak >= threshold + 18 && segment.hits >= Math.max(2, rules.minHits - 2);
        const violentSignature =
          segment.average >= threshold + 4 ||
          segment.peak >= threshold + 14 ||
          segment.averageImpact >= 28 ||
          segment.averageInstability >= 36;
        return (enoughHits && enoughDuration && violentSignature) || veryStrong;
      });
  }

  function hasConfirmedViolence() {
    computeSegments();
    return state.segments.length > 0;
  }

  function drawTimeline() {
    const { width, height } = els.timeline;
    timelineCtx.clearRect(0, 0, width, height);
    timelineCtx.fillStyle = "#fbfcfe";
    timelineCtx.fillRect(0, 0, width, height);

    timelineCtx.strokeStyle = "#d8e0e8";
    timelineCtx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
      const y = 16 + ((height - 36) * i) / 4;
      timelineCtx.beginPath();
      timelineCtx.moveTo(0, y);
      timelineCtx.lineTo(width, y);
      timelineCtx.stroke();
    }

    const threshold = getEffectiveThreshold();
    const thresholdY = height - 20 - (threshold / 100) * (height - 36);
    timelineCtx.strokeStyle = "#c5221f";
    timelineCtx.setLineDash([6, 6]);
    timelineCtx.beginPath();
    timelineCtx.moveTo(0, thresholdY);
    timelineCtx.lineTo(width, thresholdY);
    timelineCtx.stroke();
    timelineCtx.setLineDash([]);

    timelineCtx.fillStyle = "#66717f";
    timelineCtx.font = "12px system-ui, sans-serif";
    timelineCtx.fillText(`Alert ${threshold}`, 10, Math.max(14, thresholdY - 6));

    if (!state.scores.length) {
      timelineCtx.fillStyle = "#66717f";
      timelineCtx.font = "15px system-ui, sans-serif";
      timelineCtx.fillText("Risk scores will appear here", 20, height / 2);
      return;
    }

    const maxIndex = Math.max(1, state.scores.length - 1);
    timelineCtx.lineWidth = 3;
    timelineCtx.strokeStyle = "#1f6feb";
    timelineCtx.beginPath();
    state.scores.forEach((item, index) => {
      const x = (index / maxIndex) * width;
      const y = height - 20 - (item.risk / 100) * (height - 36);
      if (index === 0) {
        timelineCtx.moveTo(x, y);
      } else {
        timelineCtx.lineTo(x, y);
      }
    });
    timelineCtx.stroke();

    state.scores.forEach((item, index) => {
      if (item.risk < threshold) {
        return;
      }
      const x = (index / maxIndex) * width;
      const y = height - 20 - (item.risk / 100) * (height - 36);
      timelineCtx.fillStyle = "#c5221f";
      timelineCtx.beginPath();
      timelineCtx.arc(x, y, 4, 0, Math.PI * 2);
      timelineCtx.fill();
    });
  }

  function renderEvidence() {
    if (!state.evidence.length) {
      els.evidenceGrid.innerHTML = '<div class="empty-copy">High-risk evidence appears here</div>';
      return;
    }

    els.evidenceGrid.innerHTML = state.evidence.map((item) => `
      <article class="evidence-card">
        <img src="${item.src}" alt="Evidence frame at ${formatTime(item.time)}">
        <div>
          <span>${formatTime(item.time)}</span>
          <span class="risk-pill">${item.risk}</span>
        </div>
      </article>
    `).join("");
  }

  function renderSegments() {
    computeSegments();
    els.segmentCount.textContent = `${state.segments.length} confirmed segment${state.segments.length === 1 ? "" : "s"}`;

    if (!state.segments.length) {
      els.segmentList.innerHTML = "";
      return;
    }

    els.segmentList.innerHTML = state.segments.map((item, index) => `
      <div class="segment-item">
        <strong>Segment ${index + 1}: ${formatTime(item.start)} to ${formatTime(item.end)}</strong>
        <span>Peak ${item.peak}</span>
        <span>${item.hits} hits</span>
      </div>
    `).join("");
  }

  function renderAll() {
    const stats = getStats();
    const last = state.scores[state.scores.length - 1];
    const confirmed = hasConfirmedViolence();

    els.currentRisk.textContent = stats.current;
    els.peakRisk.textContent = stats.peak;
    els.averageRisk.textContent = stats.average;
    els.frameCount.textContent = state.scores.length;
    els.motionBar.value = last ? last.motion : 0;
    els.impactBar.value = last ? last.impact : 0;
    els.instabilityBar.value = last ? last.instability : 0;

    if (state.loadedName) {
      els.durationLabel.textContent = `${state.loadedName} | ${formatTime(state.loadedDuration)}`;
    } else if (state.scores.length) {
      els.durationLabel.textContent = `Analyzed ${formatTime(state.scores[state.scores.length - 1].time)}`;
    } else {
      els.durationLabel.textContent = "No video loaded";
    }

    els.alertBanner.classList.toggle("hidden", !confirmed);
    els.exportBtn.disabled = state.scores.length === 0;

    drawTimeline();
    renderEvidence();
    renderSegments();
  }

  function waitForEvent(target, eventName, timeoutMs) {
    return new Promise((resolve) => {
      let timer = 0;
      const done = () => {
        window.clearTimeout(timer);
        target.removeEventListener(eventName, done);
        resolve();
      };
      timer = window.setTimeout(done, timeoutMs);
      target.addEventListener(eventName, done, { once: true });
    });
  }

  async function seekVideo(time, timeoutMs) {
    const safeTime = Math.min(Math.max(0, time), Math.max(0, els.video.duration - 0.05));
    if (Math.abs(els.video.currentTime - safeTime) > 0.04) {
      els.video.currentTime = safeTime;
      await waitForEvent(els.video, "seeked", timeoutMs);
    }
    if (els.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      await waitForEvent(els.video, "loadeddata", timeoutMs);
    }
  }

  async function handleVideoFile(file) {
    stopCurrentWork();
    resetSignals();
    state.loadedName = file.name;
    state.loadedDuration = 0;
    syncPlaybackButtons();
    state.videoUrl && URL.revokeObjectURL(state.videoUrl);
    state.videoUrl = URL.createObjectURL(file);
    els.video.srcObject = null;
    els.video.src = state.videoUrl;
    els.video.controls = true;
    setMediaVisible("video");
    setStatus("Loading video", "neutral");
    await waitForEvent(els.video, "loadedmetadata", 4000);
    state.loadedDuration = Number.isFinite(els.video.duration) ? els.video.duration : 0;
    els.analyzeBtn.disabled = !state.loadedDuration;
    setStatus(state.loadedDuration ? "Video loaded" : "Unsupported video", state.loadedDuration ? "ok" : "warn");
    syncPlaybackButtons();
    renderAll();
  }

  async function analyzeLoadedVideo() {
    if (!state.loadedDuration || state.analyzing) {
      return;
    }

    stopCamera();
    resetSignals();
    setMediaVisible("video");
    state.analyzing = true;
    state.cancelRequested = false;
    els.analyzeBtn.disabled = true;
    syncPlaybackButtons();
    els.stopBtn.disabled = false;
    try {
      els.video.pause();
      els.video.muted = true;

      const duration = Math.min(state.loadedDuration, MAX_VIDEO_SECONDS);
      const profile = ANALYSIS_PROFILES[state.analysisMode] || ANALYSIS_PROFILES.balanced;
      const desiredFrames = Math.ceil(duration * profile.fps);
      const totalFrames = Math.max(1, Math.min(profile.maxFrames, desiredFrames));
      const step = duration / totalFrames;
      let processedFrames = 0;

      for (let index = 0; index < totalFrames; index += 1) {
        if (state.cancelRequested) {
          break;
        }
        const time = Math.min(index * step, duration);
        const percent = Math.max(1, Math.round(((index + 1) / totalFrames) * 100));
        setStatus(`${profile.label} analysis ${percent}% (${index + 1}/${totalFrames})`, "neutral");
        await seekVideo(time, profile.seekTimeout);

        try {
          const result = processFrame(els.video, time);
          if (result) {
            processedFrames += 1;
          }
        } catch (error) {
          console.warn("Skipped an unreadable frame", error);
        }

        await new Promise((resolve) => window.setTimeout(resolve, 0));
      }

      if (!processedFrames) {
        setStatus("Could not read video frames", "warn");
      } else {
        const confirmed = hasConfirmedViolence();
        setStatus(confirmed ? "Violence detected" : "No confirmed violence", confirmed ? "danger" : "ok");
      }
    } finally {
      state.analyzing = false;
      els.analyzeBtn.disabled = false;
      els.stopBtn.disabled = true;
      syncPlaybackButtons();
    }
  }

  async function playVideoManually() {
    if (!state.loadedDuration || state.analyzing) {
      return;
    }
    stopCamera();
    setMediaVisible("video");
    if (els.video.currentTime >= els.video.duration - 0.05) {
      els.video.currentTime = 0;
    }
    try {
      await els.video.play();
      setStatus("Playing", "ok");
    } catch (error) {
      setStatus("Playback blocked", "warn");
    }
    syncPlaybackButtons();
  }

  function pauseVideoManually() {
    if (!state.loadedDuration || state.analyzing) {
      return;
    }
    els.video.pause();
    setStatus("Paused", "neutral");
    syncPlaybackButtons();
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Camera unavailable", "warn");
      return;
    }

    stopCurrentWork();
    resetSignals();
    setMediaVisible("camera");
    state.loadedName = "Live camera";
    state.loadedDuration = 0;

    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 540 } },
        audio: false
      });
      els.video.srcObject = state.stream;
      els.video.controls = false;
      await els.video.play();
      state.liveStartedAt = performance.now();
      els.stopBtn.disabled = false;
      els.cameraBtn.disabled = true;
      setStatus("Live analysis", "ok");
      state.liveTimer = window.setInterval(() => {
        const seconds = (performance.now() - state.liveStartedAt) / 1000;
        processFrame(els.video, seconds);
      }, LIVE_INTERVAL_MS);
    } catch (error) {
      setStatus("Camera blocked", "warn");
      setMediaVisible("idle");
    }
  }

  function drawCalmFrame(frame) {
    const w = els.demoPreview.width;
    const h = els.demoPreview.height;
    demoCtx.fillStyle = "#dcecf7";
    demoCtx.fillRect(0, 0, w, h);
    demoCtx.fillStyle = "#f4d6b0";
    demoCtx.fillRect(0, h * 0.62, w, h * 0.38);
    demoCtx.fillStyle = "#3b82c4";
    demoCtx.fillRect(80 + frame * 2, 210, 110, 190);
    demoCtx.fillStyle = "#1f2937";
    demoCtx.beginPath();
    demoCtx.arc(135 + frame * 2, 175, 42, 0, Math.PI * 2);
    demoCtx.fill();
    demoCtx.fillStyle = "#15803d";
    demoCtx.fillRect(0, 405, w, 60);
    demoCtx.fillStyle = "#ffffff";
    demoCtx.font = "28px system-ui, sans-serif";
    demoCtx.fillText("Calm sample", 28, 46);
  }

  function drawRiskFrame(frame) {
    const w = els.demoPreview.width;
    const h = els.demoPreview.height;
    const shakeX = Math.sin(frame * 1.7) * 18;
    const shakeY = Math.cos(frame * 1.35) * 12;
    demoCtx.fillStyle = frame % 8 < 2 ? "#371516" : "#171b22";
    demoCtx.fillRect(0, 0, w, h);
    demoCtx.save();
    demoCtx.translate(shakeX, shakeY);
    demoCtx.fillStyle = "#334155";
    demoCtx.fillRect(0, h * 0.64, w, h * 0.36);

    const x1 = 250 + Math.sin(frame * 0.85) * 120;
    const x2 = 565 + Math.cos(frame * 0.78) * 140;
    const y1 = 145 + Math.cos(frame * 0.6) * 45;
    const y2 = 155 + Math.sin(frame * 0.7) * 50;

    demoCtx.fillStyle = "#ef4444";
    demoCtx.fillRect(x1, y1 + 80, 120, 220);
    demoCtx.fillStyle = "#f59e0b";
    demoCtx.fillRect(x2, y2 + 70, 120, 235);
    demoCtx.fillStyle = "#111827";
    demoCtx.beginPath();
    demoCtx.arc(x1 + 60, y1 + 50, 44, 0, Math.PI * 2);
    demoCtx.fill();
    demoCtx.beginPath();
    demoCtx.arc(x2 + 60, y2 + 45, 44, 0, Math.PI * 2);
    demoCtx.fill();

    demoCtx.strokeStyle = "#fef2f2";
    demoCtx.lineWidth = 12;
    demoCtx.beginPath();
    demoCtx.moveTo(x1 + 100, y1 + 130);
    demoCtx.lineTo(x2 + Math.sin(frame) * 70, y2 + 115);
    demoCtx.stroke();
    demoCtx.beginPath();
    demoCtx.moveTo(x2 + 20, y2 + 142);
    demoCtx.lineTo(x1 + Math.cos(frame) * 80, y1 + 125);
    demoCtx.stroke();

    if (frame % 5 < 2) {
      demoCtx.fillStyle = "rgba(197, 34, 31, 0.38)";
      demoCtx.fillRect(0, 0, w, h);
    }
    demoCtx.restore();

    demoCtx.fillStyle = "#ffffff";
    demoCtx.font = "28px system-ui, sans-serif";
    demoCtx.fillText("High-risk sample", 28, 46);
  }

  async function runSyntheticSample(kind) {
    stopCurrentWork();
    resetSignals();
    state.loadedName = kind === "risk" ? "High-risk sample" : "Calm sample";
    state.loadedDuration = 20;
    setMediaVisible("demo");
    els.stopBtn.disabled = false;
    setStatus("Running sample", "neutral");

    const frames = 120;
    for (let frame = 0; frame < frames; frame += 1) {
      if (state.cancelRequested) {
        break;
      }
      if (kind === "risk") {
        drawRiskFrame(frame);
      } else {
        drawCalmFrame(frame);
      }
      processFrame(els.demoPreview, frame / 6);
      await new Promise((resolve) => window.setTimeout(resolve, 24));
    }

    els.stopBtn.disabled = true;
    const confirmed = hasConfirmedViolence();
    setStatus(confirmed ? "Violence detected" : "No confirmed violence", confirmed ? "danger" : "ok");
  }

  function exportReport() {
    const stats = getStats();
    computeSegments();
    const report = {
      generatedAt: new Date().toISOString(),
      media: state.loadedName || "Untitled analysis",
      threshold: state.threshold,
      effectiveThreshold: getEffectiveThreshold(),
      sensitivity: state.sensitivityMode,
      analysisDepth: state.analysisMode,
      confirmedViolence: state.segments.length > 0,
      stats,
      segments: state.segments,
      scores: state.scores.map((item) => ({
        time: Number(item.time.toFixed(2)),
        risk: item.risk,
        motion: item.motion,
        impact: item.impact,
        instability: item.instability
      })),
      modelMode: "browser-baseline"
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `violence-report-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  els.loginForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const name = els.nameInput.value.trim();
    const email = els.emailInput.value.trim();

    if (name.length < 2) {
      els.loginError.textContent = "Enter your name.";
      els.nameInput.focus();
      return;
    }

    if (!isValidEmail(email)) {
      els.loginError.textContent = "Enter a valid email address.";
      els.emailInput.focus();
      return;
    }

    const user = { name, email };
    localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));
    showAppForUser(user);
  });

  els.logoutBtn.addEventListener("click", () => {
    localStorage.removeItem(USER_STORAGE_KEY);
    stopCurrentWork();
    showLogin();
  });

  els.uploadBtn.addEventListener("click", () => els.videoInput.click());
  els.videoInput.addEventListener("change", () => {
    const [file] = els.videoInput.files;
    if (file) {
      handleVideoFile(file);
    }
  });
  els.analyzeBtn.addEventListener("click", analyzeLoadedVideo);
  els.playBtn.addEventListener("click", playVideoManually);
  els.pauseBtn.addEventListener("click", pauseVideoManually);
  els.cameraBtn.addEventListener("click", startCamera);
  els.stopBtn.addEventListener("click", () => {
    stopCurrentWork();
    setStatus("Stopped", "warn");
  });
  els.calmSampleBtn.addEventListener("click", () => runSyntheticSample("calm"));
  els.riskSampleBtn.addEventListener("click", () => runSyntheticSample("risk"));
  els.exportBtn.addEventListener("click", exportReport);
  els.thresholdInput.addEventListener("input", () => {
    state.threshold = Number(els.thresholdInput.value);
    syncThresholdLabel();
    renderAll();
  });
  els.analysisDepth.addEventListener("change", () => {
    state.analysisMode = els.analysisDepth.value;
    const profile = ANALYSIS_PROFILES[state.analysisMode] || ANALYSIS_PROFILES.balanced;
    setStatus(`${profile.label} mode selected`, "neutral");
  });
  els.sensitivityMode.addEventListener("change", () => {
    state.sensitivityMode = els.sensitivityMode.value;
    const profile = getSensitivityProfile();
    syncThresholdLabel();
    setStatus(`${profile.label} sensitivity selected`, "neutral");
    renderAll();
  });
  els.video.addEventListener("play", syncPlaybackButtons);
  els.video.addEventListener("pause", syncPlaybackButtons);
  els.video.addEventListener("ended", syncPlaybackButtons);

  els.analysisCanvas.width = ANALYSIS_WIDTH;
  els.analysisCanvas.height = ANALYSIS_HEIGHT;
  initializeLogin();
  setMediaVisible("idle");
  syncThresholdLabel();
  syncPlaybackButtons();
  renderAll();
})();
