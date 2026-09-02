const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const errorBanner = document.getElementById("error-banner");
const results = document.getElementById("results");
const modelStatusBanner = document.getElementById("model-status-banner");
const usageStatusEl = document.getElementById("usage-status");

const predictionBadge = document.getElementById("prediction-badge");
const certaintyValue = document.getElementById("certainty-value");
const confidenceValue = document.getElementById("confidence-value");
const certaintyBanner = document.getElementById("certainty-banner");
const imgOriginal = document.getElementById("img-original");
const imgHeatmap = document.getElementById("img-heatmap");
const imgOverlay = document.getElementById("img-overlay");

const gallery = document.querySelector(".gallery");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxClose = document.querySelector(".lightbox-close");

function setSurfaceVariant(el, variant) {
  el.classList.remove("surface--sage", "surface--brick", "surface--ochre");
  el.classList.add(`surface--${variant}`);
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

// Two independent reasons the dropzone might need to be disabled: the model
// isn't loaded yet, or this IP has used up its demo uploads. Track both so
// neither one can accidentally re-enable it while the other still applies.
let modelReady = true;
let usageAvailable = true;

function updateDropzoneState() {
  fileInput.disabled = !(modelReady && usageAvailable);
}

function formatResetTime(resetAtSeconds) {
  const resetDate = new Date(resetAtSeconds * 1000);
  const diffMs = resetDate - Date.now();
  if (diffMs <= 0) return "now";
  const diffMinutes = Math.max(1, Math.ceil(diffMs / (1000 * 60)));
  if (diffMinutes < 60) {
    return `in ${diffMinutes} minute${diffMinutes === 1 ? "" : "s"}`;
  }
  const diffHours = Math.max(1, Math.ceil(diffMs / (1000 * 60 * 60)));
  if (diffHours < 24) {
    return `in ${diffHours} hour${diffHours === 1 ? "" : "s"}`;
  }
  return `on ${resetDate.toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" })}`;
}

function renderUsage(usage) {
  if (typeof usage.remaining !== "number") return;

  usageStatusEl.hidden = false;
  if (usage.remaining > 0) {
    usageStatusEl.textContent = `${usage.remaining} of ${usage.limit ?? usage.remaining} demo uploads remaining`;
    usageAvailable = true;
  } else {
    usageStatusEl.textContent = `Demo limit reached — resets ${formatResetTime(usage.reset_at)}`;
    usageAvailable = false;
  }
  updateDropzoneState();
}

async function checkUsageStatus() {
  try {
    const response = await fetch("/api/usage");
    const data = await response.json();
    renderUsage(data);
  } catch (err) {
    // Non-critical -- the limit is still enforced server-side even if this
    // status display fails to load.
  }
}

async function checkModelStatus() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();

    if (data.model_loaded) {
      modelStatusBanner.hidden = true;
      modelReady = true;
      updateDropzoneState();
      return;
    }

    if (data.error) {
      modelStatusBanner.textContent = "The model failed to load. Please try again later.";
      setSurfaceVariant(modelStatusBanner, "brick");
      modelStatusBanner.hidden = false;
      modelReady = false;
      updateDropzoneState();
      return;
    }

    modelStatusBanner.textContent = "Model is warming up — this can take up to a minute on a cold start.";
    setSurfaceVariant(modelStatusBanner, "ochre");
    modelStatusBanner.hidden = false;
    modelReady = false;
    updateDropzoneState();
    setTimeout(checkModelStatus, 3000);
  } catch (err) {
    setTimeout(checkModelStatus, 3000);
  }
}

function renderResults(data) {
  predictionBadge.innerHTML = `<strong>Prediction: ${data.prediction}</strong>`;
  setSurfaceVariant(predictionBadge, data.prediction === "Malignant" ? "brick" : "sage");

  certaintyValue.textContent = formatPercent(data.certainty);
  confidenceValue.textContent = formatPercent(data.confidence);

  let certaintyText;
  let certaintyVariant;
  if (data.certainty > 0.6) {
    certaintyText = "High certainty prediction";
    certaintyVariant = "sage";
  } else if (data.certainty > 0.2) {
    certaintyText = "Moderate certainty";
    certaintyVariant = "ochre";
  } else {
    certaintyText = "Low certainty - uncertain prediction";
    certaintyVariant = "brick";
  }
  certaintyBanner.textContent = certaintyText;
  setSurfaceVariant(certaintyBanner, certaintyVariant);

  imgOriginal.src = data.images.original;
  imgHeatmap.src = data.images.heatmap;
  imgOverlay.src = data.images.overlay;

  results.hidden = false;
  renderUsage(data);
}

async function analyzeFile(file) {
  clearError();
  results.hidden = true;
  statusEl.hidden = false;
  fileInput.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.detail || "Something went wrong while analyzing the image.");
      renderUsage(data);
      return;
    }

    renderResults(data);
  } catch (err) {
    showError("Could not reach the server. Please try again.");
  } finally {
    statusEl.hidden = true;
    updateDropzoneState();
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    analyzeFile(file);
  }
});

function openLightbox(src, alt) {
  lightboxImg.src = src;
  lightboxImg.alt = alt || "";
  lightbox.hidden = false;
}

function closeLightbox() {
  lightbox.hidden = true;
  lightboxImg.src = "";
}

gallery.addEventListener("click", (event) => {
  const img = event.target.closest("img");
  if (img) {
    openLightbox(img.src, img.alt);
  }
});

lightboxClose.addEventListener("click", closeLightbox);

lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) {
    closeLightbox();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !lightbox.hidden) {
    closeLightbox();
  }
});

checkModelStatus();
checkUsageStatus();
