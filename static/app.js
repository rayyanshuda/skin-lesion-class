const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const errorBanner = document.getElementById("error-banner");
const results = document.getElementById("results");

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
      return;
    }

    renderResults(data);
  } catch (err) {
    showError("Could not reach the server. Please try again.");
  } finally {
    statusEl.hidden = true;
    fileInput.disabled = false;
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
