
const xrayInput = document.getElementById("xrayInput");
const uploadTitle = document.getElementById("uploadTitle");
const uploadSubtitle = document.getElementById("uploadSubtitle");
const analyzeButton = document.getElementById("analyzeButton");
const loadingState = document.getElementById("loadingState");
const errorMessage = document.getElementById("errorMessage");
const resultsSection = document.getElementById("resultsSection");
const xrayPreview = document.getElementById("xrayPreview");
const flaggedFindings = document.getElementById("flaggedFindings");
const noFindingCard = document.getElementById("noFindingCard");
const resultSummaryBadge = document.getElementById("resultSummaryBadge");
const explanationSection = document.getElementById("explanationSection");
const findingSelect = document.getElementById("findingSelect");
const explainButton = document.getElementById("explainButton");
const gradcamEmpty = document.getElementById("gradcamEmpty");
const gradcamResult = document.getElementById("gradcamResult");
const gradcamImage = document.getElementById("gradcamImage");
const toggleScoresButton = document.getElementById("toggleScoresButton");
const toggleScoresLabel = document.getElementById("toggleScoresLabel");
const toggleScoresIcon = document.getElementById("toggleScoresIcon");
const allScoresContainer = document.getElementById("allScoresContainer");
const scoresTableBody = document.getElementById("scoresTableBody");
const downloadReportButton = document.getElementById("downloadReportButton");

const devButton = document.getElementById("devButton");
const devDrawer = document.getElementById("devDrawer");
const devOverlay = document.getElementById("devOverlay");
const closeDevButton = document.getElementById("closeDevButton");
const devMode = document.getElementById("devMode");
const devApiStatus = document.getElementById("devApiStatus");
const devModel = document.getElementById("devModel");
const devModelVersion = document.getElementById("devModelVersion");
const devPredictionCards = document.getElementById("devPredictionCards");

let selectedFile = null;
let latestPredictionResponse = null;
let latestUploadedFileName = null;
let currentPreviewUrl = null;

devMode.textContent = window.APP_CONFIG.MOCK_MODE ? "Mock" : "Backend";
devModel.textContent = window.APP_CONFIG.MODEL_NAME;
devApiStatus.textContent = window.APP_CONFIG.MOCK_MODE ? "Mock response" : "Waiting";

function formatPercentage(value) {
    return `${Math.round(value * 100)}%`;
}

function formatDecimal(value) {
    return Number(value).toFixed(2);
}

function displayName(name) {
    return name.replaceAll("_", " ");
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}

function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}

function setLoading(isLoading) {
    if (isLoading) {
        loadingState.classList.remove("hidden");
        analyzeButton.disabled = true;
    } else {
        loadingState.classList.add("hidden");
        analyzeButton.disabled = !selectedFile;
    }
}

function sortByProbabilityDesc(items) {
    return [...items].sort((a, b) => b.probability - a.probability);
}

function clearPreviewUrl() {
    if (currentPreviewUrl) {
        URL.revokeObjectURL(currentPreviewUrl);
        currentPreviewUrl = null;
    }
}

function showFilePreview(file) {
    clearPreviewUrl();

    currentPreviewUrl = URL.createObjectURL(file);
    xrayPreview.src = currentPreviewUrl;
    xrayPreview.classList.remove("hidden");
}

xrayInput.addEventListener("change", () => {
    clearError();

    const file = xrayInput.files[0];

    if (!file) {
        selectedFile = null;
        latestUploadedFileName = null;
        analyzeButton.disabled = true;
        return;
    }

    const allowedTypes = [
        "image/png",
        "image/jpeg"
    ];

    if (!allowedTypes.includes(file.type)) {
        selectedFile = null;
        latestUploadedFileName = null;
        xrayInput.value = "";
        analyzeButton.disabled = true;

        showError(
            "Unsupported file format. Please upload a PNG, JPG, or JPEG chest X-ray."
        );
        return;
    }

    selectedFile = file;
    latestUploadedFileName = file.name;

    uploadTitle.textContent = file.name;
    uploadSubtitle.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · Image`;

    analyzeButton.disabled = false;
    showFilePreview(file);
});

analyzeButton.addEventListener("click", async () => {
    if (!selectedFile) return;

    clearError();
    resultsSection.classList.add("hidden");

    gradcamResult.classList.add("hidden");
    gradcamEmpty.classList.remove("hidden");

    setLoading(true);

    try {
        const data = await window.TeamCoreAPI.predictXray(selectedFile);

        latestPredictionResponse = data;

        renderPredictionResults(data);
        renderDeveloperCards(data.predictions);

        devModelVersion.textContent = data.model_version || "Not provided";
        devApiStatus.textContent = window.APP_CONFIG.MOCK_MODE ? "Mock response valid" : "Connected";

        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        console.error(error);

        devApiStatus.textContent = "Error";

        showError(
            window.APP_CONFIG.MOCK_MODE
                ? error.message
                : "The inference service is temporarily unavailable. Please try again."
        );
    } finally {
        setLoading(false);
    }
});

function renderPredictionResults(data) {
    const positives = sortByProbabilityDesc(
        data.predictions.filter((prediction) => prediction.positive)
    );

    const allSorted = sortByProbabilityDesc(data.predictions);

    renderSummary(positives);
    renderFlaggedFindings(positives);
    renderFindingSelector(positives);
    renderAllScores(allSorted);
}

function renderSummary(positives) {
    const count = positives.length;

    if (count === 0) {
        resultSummaryBadge.textContent = "No findings flagged";
        return;
    }

    resultSummaryBadge.textContent = `${count} finding${count === 1 ? "" : "s"} flagged`;
}

function renderFlaggedFindings(positives) {
    flaggedFindings.innerHTML = "";

    if (positives.length === 0) {
        flaggedFindings.classList.add("hidden");
        noFindingCard.classList.remove("hidden");
        return;
    }

    flaggedFindings.classList.remove("hidden");
    noFindingCard.classList.add("hidden");

    positives.forEach((prediction, index) => {
        const card = document.createElement("article");
        const variantClass = index === 0 ? "finding-card-primary" : "finding-card-secondary";

        card.className = `finding-card ${variantClass}`;
        card.setAttribute("data-pathology", prediction.name);

        card.innerHTML = `
            <div class="finding-name">
                <strong>${displayName(prediction.name)}</strong>
                <span>
                    ${index === 0
                        ? "Highest model score among flagged findings"
                        : "Flagged by the model"}
                </span>
            </div>

            <div class="finding-meta">
                ${index === 0 ? `<span class="top-score-tag">Top score</span>` : ""}

                <div class="finding-score-wrap">
                    <span class="finding-score-label">Model score</span>
                    <div class="finding-score">${formatPercentage(prediction.probability)}</div>
                </div>
            </div>
        `;

        card.addEventListener("click", () => {
            findingSelect.value = prediction.name;

            if (!window.APP_CONFIG.MOCK_MODE) {
                updateGradcamForSelectedFinding();
            }
        });

        flaggedFindings.appendChild(card);
    });
}

function renderFindingSelector(positives) {
    findingSelect.innerHTML = "";

    if (positives.length === 0) {
        explanationSection.classList.add("hidden");
        return;
    }

    explanationSection.classList.remove("hidden");

    positives.forEach((prediction) => {
        const option = document.createElement("option");
        option.value = prediction.name;
        option.textContent = `${displayName(prediction.name)} · ${formatPercentage(prediction.probability)}`;
        findingSelect.appendChild(option);
    });

    findingSelect.value = positives[0].name;
}

function renderAllScores(predictions) {
    scoresTableBody.innerHTML = "";

    predictions.forEach((prediction) => {
        const row = document.createElement("tr");
        const statusClass = prediction.positive ? "status-flagged" : "status-below";
        const statusText = prediction.positive ? "Flagged" : "Below threshold";

        if (prediction.positive) {
            row.classList.add("score-row-flagged");
        }

        row.innerHTML = `
            <td>${displayName(prediction.name)}</td>
            <td>${formatPercentage(prediction.probability)}</td>
            <td class="${statusClass}">${statusText}</td>
        `;

        scoresTableBody.appendChild(row);
    });
}

function renderDeveloperCards(predictions) {
    const sorted = sortByProbabilityDesc(predictions);
    devPredictionCards.innerHTML = "";

    sorted.forEach((prediction) => {
        const delta = prediction.probability - prediction.threshold;

        const card = document.createElement("article");
        card.className = `dev-prediction-card ${prediction.positive ? "flagged" : "not-flagged"}`;

        card.innerHTML = `
            <div class="dev-prediction-top">
                <div class="dev-prediction-name">${displayName(prediction.name)}</div>
                <span class="dev-prediction-status ${prediction.positive ? "flagged" : "below"}">
                    ${prediction.positive ? "Flagged" : "Below"}
                </span>
            </div>

            <div class="dev-metrics">
                <div class="dev-metric">
                    <span>Score</span>
                    <strong>${formatDecimal(prediction.probability)}</strong>
                </div>

                <div class="dev-metric">
                    <span>Threshold</span>
                    <strong>${formatDecimal(prediction.threshold)}</strong>
                </div>

                <div class="dev-metric">
                    <span>Delta</span>
                    <strong>${delta >= 0 ? "+" : ""}${formatDecimal(delta)}</strong>
                </div>
            </div>
        `;

        devPredictionCards.appendChild(card);
    });
}

toggleScoresButton.addEventListener("click", () => {
    const isHidden = allScoresContainer.classList.contains("hidden");

    allScoresContainer.classList.toggle("hidden");
    toggleScoresButton.setAttribute("aria-expanded", isHidden ? "true" : "false");

    toggleScoresLabel.textContent = isHidden
        ? "Hide all 14 model scores"
        : "View all 14 model scores";
});

async function updateGradcamForSelectedFinding() {
    if (!selectedFile || window.APP_CONFIG.MOCK_MODE) return;

    const pathology = findingSelect.value;
    if (!pathology) return;

    clearError();

    explainButton.disabled = true;
    explainButton.textContent = "Updating…";

    try {
        const imageBlob = await window.TeamCoreAPI.requestGradcam(selectedFile, pathology);
        const imageUrl = URL.createObjectURL(imageBlob);

        gradcamImage.src = imageUrl;
        gradcamEmpty.classList.add("hidden");
        gradcamResult.classList.remove("hidden");
    } catch (error) {
        console.error(error);
        showError("The visual explanation could not be generated. Please try again.");
    } finally {
        explainButton.disabled = false;
        explainButton.textContent = "Update Explanation";
    }
}

explainButton.addEventListener("click", async () => {
    if (!selectedFile) return;

    if (window.APP_CONFIG.MOCK_MODE) {
        showError(
            "Grad-CAM is still disconnected in Mock Mode. In the integrated version, the highest flagged finding will be selected first, and this control will switch the real heatmap between flagged findings."
        );
        return;
    }

    await updateGradcamForSelectedFinding();
});

findingSelect.addEventListener("change", () => {
    if (!window.APP_CONFIG.MOCK_MODE) {
        updateGradcamForSelectedFinding();
    }
});

downloadReportButton.addEventListener("click", () => {
    if (!latestPredictionResponse) {
        showError("Run an analysis first to download the summary report.");
        return;
    }

    const sorted = sortByProbabilityDesc(latestPredictionResponse.predictions);
    const flagged = sorted.filter((item) => item.positive);

    const lines = [
        "TEAM CORE — SUMMARY REPORT",
        "Samsung Innovation Campus · Misk",
        "",
        `Model: ${window.APP_CONFIG.MODEL_NAME}`,
        `Model version: ${latestPredictionResponse.model_version || "Not provided"}`,
        `Frontend mode: ${window.APP_CONFIG.MOCK_MODE ? "Mock" : "Backend"}`,
        `Uploaded file: ${latestUploadedFileName || "Unknown"}`,
        "",
        "FLAGGED FINDINGS",
        flagged.length
            ? flagged.map((item, index) => `${index + 1}. ${displayName(item.name)} — ${formatPercentage(item.probability)}`).join("\\n")
            : "No findings were flagged by the model.",
        "",
        "ALL MODEL SCORES (SORTED)",
        ...sorted.map((item) => `${displayName(item.name)} | score ${formatPercentage(item.probability)} | threshold ${formatDecimal(item.threshold)} | ${item.positive ? "Flagged" : "Below threshold"}`),
        "",
        "DISCLAIMER",
        "Research and educational prototype. Results are model-generated probabilities and should not be interpreted as a clinical diagnosis."
    ];

    const blob = new Blob([lines.join("\\n")], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "team-core-summary-report.txt";
    link.click();

    URL.revokeObjectURL(url);
});

function openDevDrawer() {
    devDrawer.classList.add("open");
    devDrawer.setAttribute("aria-hidden", "false");
    devOverlay.classList.remove("hidden");
}

function closeDevDrawer() {
    devDrawer.classList.remove("open");
    devDrawer.setAttribute("aria-hidden", "true");
    devOverlay.classList.add("hidden");
}

devButton.addEventListener("click", openDevDrawer);
closeDevButton.addEventListener("click", closeDevDrawer);
devOverlay.addEventListener("click", closeDevDrawer);

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        closeDevDrawer();
    }
});

