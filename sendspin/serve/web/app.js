/**
 * Sendspin Embedded Player
 * Auto-connects to the server that serves this page.
 */

const MAX_VOLUME = 100;
const UI_ACTIVATION_MS = 550;
const SYNC_UPDATE_INTERVAL_MS = 250;
const COPY_FEEDBACK_MS = 2000;
const LISTENER_POLL_INTERVAL_MS = 10_000;
const START_HAPTIC_PATTERN = [18, 28, 24];
const STOP_HAPTIC_PATTERN = [14];
const SYNC_PLACEHOLDER = "--.- ms";
const SYNC_CLASSES = ["sync-good", "sync-warn", "sync-bad", "sync-idle"];
const SYNC_GRAPH = {
  rangeMs: 50,
  historyLength: 180,
  sampleIntervalMs: 45,
  insets: {
    left: 0,
    right: 18,
    top: 6,
    bottom: 6,
  },
  labels: {
    xInset: 6,
    positiveOffsetY: 12,
    zeroOffsetY: 1,
    negativeOffsetY: -12,
  },
  strokeWidthPx: 2.5,
  endpointRadiusPx: 4.5,
  lineShadowBlurPx: 16,
  pointShadowBlurPx: 14,
};
const TONE_COLORS = {
  "sync-idle": [239, 225, 187],
  "sync-good": [245, 255, 246],
  "sync-warn": [255, 224, 130],
  "sync-bad": [255, 154, 146],
};

// DOM elements
const elements = {
  body: document.body,
  controlCard: document.getElementById("control-card"),
  listenToggleBtn: document.getElementById("listen-toggle-btn"),
  syncPanel: document.getElementById("sync-panel"),
  syncStatus: document.getElementById("sync-status"),
  syncGraphShell: document.getElementById("sync-graph-shell"),
  syncGraph: document.getElementById("sync-graph"),
  shareCard: document.getElementById("share-card"),
  qrCode: document.getElementById("qr-code"),
  shareBtn: document.getElementById("share-btn"),
  shareServerUrl: document.getElementById("share-server-url"),
  castLink: document.getElementById("cast-link"),
  listenerCount: document.getElementById("listener-count"),
  listenerCountValue: document.getElementById("listener-count-value"),
};

const state = {
  player: null,
  syncUpdateInterval: null,
  isListening: false,
  isStarting: false,
  showPostAnimationLabel: false,
  listenerPollInterval: null,
  sync: {
    currentMs: null,
    tone: "sync-idle",
  },
};

// Auto-derive server URL from current page location
const serverUrl = `${location.protocol}//${location.host}`;
elements.shareServerUrl.textContent = serverUrl;
elements.shareServerUrl.href = serverUrl;

function wait(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function triggerHaptic(pattern) {
  if (typeof navigator.vibrate !== "function") {
    return;
  }

  try {
    navigator.vibrate(pattern);
  } catch (err) {
    console.warn("Failed to trigger vibration:", err);
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function rgba(color, alpha) {
  const [red, green, blue] = color;
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function applySyncToneClass(element, tone) {
  element.classList.remove(...SYNC_CLASSES);
  element.classList.add(tone);
}

function formatSyncValue(syncMs) {
  const normalizedSyncMs = Math.abs(syncMs) < 0.05 ? 0 : syncMs;
  return `${normalizedSyncMs.toFixed(1)} ms`;
}

function getSyncTone(syncMs) {
  const absSyncMs = Math.abs(syncMs);
  if (absSyncMs < 10) {
    return "sync-good";
  }
  if (absSyncMs <= 25) {
    return "sync-warn";
  }
  return "sync-bad";
}

function createSyncGraph({ canvas, shell }) {
  let animationFrame = null;
  let lastSampleAtMs = 0;
  let history = [];
  let currentSyncMs = null;
  let currentTone = "sync-idle";
  const resetThresholdMs =
    SYNC_GRAPH.sampleIntervalMs * SYNC_GRAPH.historyLength;

  function clearHistory() {
    history = [];
    lastSampleAtMs = 0;
  }

  function getContext() {
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return null;
    }

    const dpr = window.devicePixelRatio || 1;
    const width = rect.width;
    const height = rect.height;
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);

    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return null;
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function getMetrics(width, height) {
    const { left, right, top, bottom } = SYNC_GRAPH.insets;
    return {
      width,
      height,
      left,
      right,
      top,
      bottom,
      plotWidth: width - left - right,
      plotHeight: height - top - bottom,
    };
  }

  function getX(index, historyLength, metrics) {
    const ageFromNewest = historyLength - 1 - index;
    const ratio = ageFromNewest / Math.max(SYNC_GRAPH.historyLength - 1, 1);
    return metrics.width - metrics.right - ratio * metrics.plotWidth;
  }

  function getY(syncMs, metrics) {
    const clamped = clamp(syncMs, -SYNC_GRAPH.rangeMs, SYNC_GRAPH.rangeMs);
    const ratio = (SYNC_GRAPH.rangeMs - clamped) / (SYNC_GRAPH.rangeMs * 2);
    return metrics.top + ratio * metrics.plotHeight;
  }

  function getGridLines() {
    const max = SYNC_GRAPH.rangeMs;
    const half = max / 2;
    return [
      { value: max, label: String(max), alpha: 0.16, dash: [] },
      { value: half, label: null, alpha: 0.08, dash: [4, 6] },
      { value: 0, label: "0", alpha: 0.22, dash: [] },
      { value: -half, label: null, alpha: 0.08, dash: [4, 6] },
      { value: -max, label: String(-max), alpha: 0.16, dash: [] },
    ];
  }

  function getLabelOffsetY(value) {
    if (value > 0) {
      return SYNC_GRAPH.labels.positiveOffsetY;
    }
    if (value < 0) {
      return SYNC_GRAPH.labels.negativeOffsetY;
    }
    return SYNC_GRAPH.labels.zeroOffsetY;
  }

  function drawGrid(ctx, metrics) {
    const lines = getGridLines();

    ctx.save();
    ctx.font = '11px "SF Mono", "Monaco", "Menlo", monospace';
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (const line of lines) {
      const y = getY(line.value, metrics);
      ctx.beginPath();
      ctx.setLineDash(line.dash);
      ctx.moveTo(metrics.left, y);
      ctx.lineTo(metrics.width - metrics.right, y);
      ctx.strokeStyle = `rgba(255, 255, 255, ${line.alpha})`;
      ctx.lineWidth = line.value === 0 ? 1.2 : 1;
      ctx.stroke();

      if (line.label !== null) {
        const labelY = clamp(
          y + getLabelOffsetY(line.value),
          12,
          metrics.height - 12,
        );
        ctx.fillStyle = "rgba(255, 255, 255, 0.46)";
        ctx.fillText(line.label, metrics.width - SYNC_GRAPH.labels.xInset, labelY);
      }
    }

    ctx.restore();
  }

  function traceSmoothLine(ctx, points) {
    if (points.length === 0) {
      return;
    }

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    if (points.length === 1) {
      return;
    }

    for (let i = 1; i < points.length - 1; i += 1) {
      const midX = (points[i].x + points[i + 1].x) / 2;
      const midY = (points[i].y + points[i + 1].y) / 2;
      ctx.quadraticCurveTo(points[i].x, points[i].y, midX, midY);
    }

    const lastPoint = points[points.length - 1];
    ctx.quadraticCurveTo(lastPoint.x, lastPoint.y, lastPoint.x, lastPoint.y);
  }

  function getSegments(metrics) {
    const segments = [];
    let currentSegment = [];

    for (let i = 0; i < history.length; i += 1) {
      const sample = history[i];
      if (typeof sample.syncMs !== "number") {
        if (currentSegment.length > 0) {
          segments.push(currentSegment);
          currentSegment = [];
        }
        continue;
      }

      currentSegment.push({
        x: getX(i, history.length, metrics),
        y: getY(sample.syncMs, metrics),
      });
    }

    if (currentSegment.length > 0) {
      segments.push(currentSegment);
    }

    return segments;
  }

  function drawLine(ctx, metrics) {
    const segments = getSegments(metrics);
    if (segments.length === 0) {
      return;
    }

    const toneColor = TONE_COLORS[currentTone] ?? TONE_COLORS["sync-idle"];
    const strokeGradient = ctx.createLinearGradient(
      metrics.left,
      0,
      metrics.width - metrics.right,
      0,
    );
    strokeGradient.addColorStop(0, rgba(toneColor, 0.12));
    strokeGradient.addColorStop(0.7, rgba(toneColor, 0.58));
    strokeGradient.addColorStop(1, rgba(toneColor, 0.98));

    ctx.save();
    ctx.lineWidth = SYNC_GRAPH.strokeWidthPx;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = strokeGradient;
    ctx.shadowBlur = SYNC_GRAPH.lineShadowBlurPx;
    ctx.shadowColor = rgba(toneColor, 0.28);

    for (const segment of segments) {
      traceSmoothLine(ctx, segment);
      ctx.stroke();
    }

    ctx.restore();

    const lastSegment = segments[segments.length - 1];
    const lastPoint = lastSegment[lastSegment.length - 1];
    if (!lastPoint) {
      return;
    }

    ctx.save();
    ctx.beginPath();
    ctx.arc(lastPoint.x, lastPoint.y, SYNC_GRAPH.endpointRadiusPx, 0, Math.PI * 2);
    ctx.fillStyle = rgba(toneColor, 0.98);
    ctx.shadowBlur = SYNC_GRAPH.pointShadowBlurPx;
    ctx.shadowColor = rgba(toneColor, 0.42);
    ctx.fill();
    ctx.restore();
  }

  function draw() {
    const graph = getContext();
    if (!graph) {
      return;
    }

    const { ctx, width, height } = graph;
    const metrics = getMetrics(width, height);

    ctx.clearRect(0, 0, width, height);
    drawGrid(ctx, metrics);
    drawLine(ctx, metrics);
  }

  function sampleHistory() {
    history.push({ syncMs: currentSyncMs });

    if (history.length > SYNC_GRAPH.historyLength) {
      history.shift();
    }
  }

  function loop(timestampMs) {
    if (lastSampleAtMs === 0) {
      lastSampleAtMs = timestampMs;
      sampleHistory();
    }

    const elapsedMs = timestampMs - lastSampleAtMs;

    // After long tab suspension, drop stale history instead of replaying it.
    if (elapsedMs > resetThresholdMs) {
      clearHistory();
      lastSampleAtMs = timestampMs;
      sampleHistory();
    } else {
      while (timestampMs - lastSampleAtMs >= SYNC_GRAPH.sampleIntervalMs) {
        lastSampleAtMs += SYNC_GRAPH.sampleIntervalMs;
        sampleHistory();
      }
    }

    draw();
    animationFrame = window.requestAnimationFrame(loop);
  }

  return {
    start() {
      if (animationFrame !== null) {
        return;
      }

      draw();
      animationFrame = window.requestAnimationFrame(loop);
    },

    stop() {
      if (animationFrame === null) {
        return;
      }

      window.cancelAnimationFrame(animationFrame);
      animationFrame = null;
    },

    reset() {
      clearHistory();
      draw();
    },

    updateSample({ syncMs, tone }) {
      currentSyncMs = syncMs;
      currentTone = tone;
      applySyncToneClass(shell, tone);

      if (animationFrame === null) {
        draw();
      }
    },
  };
}

const syncGraph = createSyncGraph({
  canvas: elements.syncGraph,
  shell: elements.syncGraphShell,
});

function renderSyncDisplay({ label, tone = "sync-idle", syncMs = null }) {
  state.sync.currentMs = syncMs;
  state.sync.tone = tone;
  elements.syncStatus.textContent = label;
  applySyncToneClass(elements.syncStatus, tone);
  syncGraph.updateSample({ syncMs, tone });
}

function resetSyncDisplay() {
  renderSyncDisplay({
    label: SYNC_PLACEHOLDER,
    tone: "sync-idle",
    syncMs: null,
  });
}

function getListenToggleLabel() {
  if (state.isStarting && state.showPostAnimationLabel) {
    return "Connecting...";
  }

  return state.isListening ? "Stop Listening" : "Start Listening";
}

function renderUiState() {
  const pageIsActive = state.isListening || state.isStarting;

  elements.body.classList.toggle("is-listening", pageIsActive);
  elements.body.classList.toggle("is-starting", state.isStarting);
  elements.controlCard.classList.toggle("is-expanded", pageIsActive);
  elements.syncPanel.setAttribute("aria-hidden", String(!pageIsActive));
  elements.listenToggleBtn.setAttribute("aria-pressed", String(pageIsActive));
  elements.listenToggleBtn.textContent = getListenToggleLabel();
}

function handlePlayerStateChange() {
  if (!state.player) {
    return;
  }

  updateSyncStatus();
}

function stopSyncUpdates() {
  if (state.syncUpdateInterval === null) {
    return;
  }

  window.clearInterval(state.syncUpdateInterval);
  state.syncUpdateInterval = null;
}

function startSyncUpdates() {
  stopSyncUpdates();
  state.syncUpdateInterval = window.setInterval(
    updateSyncStatus,
    SYNC_UPDATE_INTERVAL_MS,
  );
}

function destroyPlayer(reason = "shutdown") {
  stopSyncUpdates();

  if (!state.player) {
    return;
  }

  const activePlayer = state.player;
  state.player = null;

  try {
    activePlayer.disconnect(reason);
  } catch (err) {
    console.warn("Failed to disconnect player:", err);
  }
}

async function createPlayer() {
  const { SendspinPlayer } = await sdkImport;
  return new SendspinPlayer({
    baseUrl: serverUrl,
    onStateChange: handlePlayerStateChange,
  });
}

async function connectPlayer() {
  if (state.player?.isConnected) {
    return state.player;
  }

  destroyPlayer("user_request");

  const nextPlayer = await createPlayer();
  state.player = nextPlayer;

  try {
    await nextPlayer.connect();
    startSyncUpdates();
    return nextPlayer;
  } catch (err) {
    if (state.player === nextPlayer) {
      destroyPlayer("user_request");
    } else {
      try {
        nextPlayer.disconnect("user_request");
      } catch (disconnectErr) {
        console.warn("Failed to clean up after connection error:", disconnectErr);
      }
    }

    throw err;
  }
}

function updateSyncStatus() {
  const activePlayer = state.player;
  if (!activePlayer) {
    return;
  }

  if (!activePlayer.isConnected) {
    disconnect();
    return;
  }

  const syncInfo = activePlayer.syncInfo ?? {};
  const syncMs =
    typeof syncInfo.syncErrorMs === "number" &&
    Number.isFinite(syncInfo.syncErrorMs)
      ? syncInfo.syncErrorMs
      : null;

  if (!activePlayer.isPlaying || syncMs === null) {
    resetSyncDisplay();
    return;
  }

  renderSyncDisplay({
    label: formatSyncValue(syncMs),
    tone: getSyncTone(syncMs),
    syncMs,
  });
}

async function startListening() {
  if (state.isListening || state.isStarting) {
    return;
  }

  state.isListening = true;
  state.isStarting = true;
  state.showPostAnimationLabel = false;
  elements.listenToggleBtn.disabled = true;

  resetSyncDisplay();
  syncGraph.reset();
  syncGraph.start();
  startListenerPolling();
  renderUiState();

  try {
    const connectPromise = connectPlayer();

    await wait(UI_ACTIVATION_MS);
    state.showPostAnimationLabel = true;
    renderUiState();

    const activePlayer = await connectPromise;
    activePlayer.setVolume(MAX_VOLUME);
    activePlayer.setMuted(false);
    updateSyncStatus();
  } catch (err) {
    console.error("Connection failed:", err);
    disconnect();
  } finally {
    state.isStarting = false;
    state.showPostAnimationLabel = false;
    elements.listenToggleBtn.disabled = false;
    renderUiState();
  }
}

function stopListening() {
  disconnect("user_request");
}

function disconnect(reason = "shutdown") {
  destroyPlayer(reason);

  state.isListening = false;
  state.isStarting = false;
  state.showPostAnimationLabel = false;
  elements.listenToggleBtn.disabled = false;

  syncGraph.stop();
  stopListenerPolling();
  resetSyncDisplay();
  syncGraph.reset();
  renderUiState();
}

// Set up Cast link with server URL
elements.castLink.href = `https://sendspin.github.io/cast/?host=${encodeURIComponent(
  serverUrl,
)}`;

if (["localhost", "127.0.0.1"].includes(location.hostname)) {
  elements.shareCard.textContent = "Sharing disabled when visiting localhost";
}

elements.listenToggleBtn.addEventListener("click", async () => {
  if (state.isListening) {
    triggerHaptic(STOP_HAPTIC_PATTERN);
    stopListening();
    return;
  }

  triggerHaptic(START_HAPTIC_PATTERN);
  await startListening();
});

const sdkImport = import(
  "https://cdn.jsdelivr.net/npm/@sendspin/sendspin-js@5.0.0/+esm",
);

// QR Code generation (using qrcode-generator loaded via script tag)
if (typeof qrcode !== "undefined") {
  const qr = qrcode(0, "M");
  qr.addData(location.href);
  qr.make();
  elements.qrCode.innerHTML = qr.createSvgTag({ cellSize: 4, margin: 2 });
}

// Share button - copy URL to clipboard
elements.shareBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(location.href);
  } catch (err) {
    // Fallback for browsers without clipboard API
    const textArea = document.createElement("textarea");
    textArea.value = location.href;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand("copy");
    document.body.removeChild(textArea);
  }

  const originalText = elements.shareBtn.textContent;
  elements.shareBtn.textContent = "Copied!";
  window.setTimeout(() => {
    elements.shareBtn.textContent = originalText;
  }, COPY_FEEDBACK_MS);
});

// Listener count polling
async function updateListenerCount() {
  try {
    const resp = await fetch(`${serverUrl}/api/status`);
    if (resp.ok) {
      const data = await resp.json();
      const count = Math.max(data.total_clients ?? 0, 1);
      elements.listenerCountValue.textContent = String(count);
      elements.listenerCount.setAttribute("aria-hidden", "false");
    }
  } catch {
    // Silently ignore - server may be unavailable
  }
}

function startListenerPolling() {
  updateListenerCount();
  state.listenerPollInterval = window.setInterval(
    updateListenerCount,
    LISTENER_POLL_INTERVAL_MS,
  );
}

function stopListenerPolling() {
  if (state.listenerPollInterval !== null) {
    window.clearInterval(state.listenerPollInterval);
    state.listenerPollInterval = null;
  }
  elements.listenerCount.setAttribute("aria-hidden", "true");
}

renderUiState();
resetSyncDisplay();
