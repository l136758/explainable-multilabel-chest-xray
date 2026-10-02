/**
 * Team Core Frontend API Client
 *
 * During frontend development:
 *     MOCK_MODE = true
 *
 * During integration:
 *     MOCK_MODE = false
 */

async function delay(milliseconds) {
    return new Promise((resolve) => {
        setTimeout(resolve, milliseconds);
    });
}

function validatePredictionResponse(data) {
    if (!data || typeof data !== "object") {
        throw new Error("Prediction response is not a valid object.");
    }

    if (!Array.isArray(data.predictions)) {
        throw new Error("Prediction response is missing predictions.");
    }

    if (data.predictions.length !== 14) {
        throw new Error(
            `Expected 14 predictions but received ${data.predictions.length}.`
        );
    }

    const expectedLabels = window.APP_CONFIG.LABELS;

    data.predictions.forEach((prediction, index) => {
        const expectedName = expectedLabels[index];

        if (prediction.name !== expectedName) {
            throw new Error(
                `Unexpected label order at index ${index}. ` +
                `Expected "${expectedName}" but received "${prediction.name}".`
            );
        }

        if (
            typeof prediction.probability !== "number" ||
            prediction.probability < 0 ||
            prediction.probability > 1
        ) {
            throw new Error(
                `Invalid probability for ${prediction.name}.`
            );
        }

        if (
            typeof prediction.threshold !== "number" ||
            prediction.threshold < 0 ||
            prediction.threshold > 1
        ) {
            throw new Error(
                `Invalid threshold for ${prediction.name}.`
            );
        }

        if (typeof prediction.positive !== "boolean") {
            throw new Error(
                `Invalid positive flag for ${prediction.name}.`
            );
        }
    });

    return data;
}

async function getMockPrediction() {
    await delay(900);

    const response = await fetch("mocks/prediction.json");

    if (!response.ok) {
        throw new Error(
            "Could not load the mock prediction response."
        );
    }

    const data = await response.json();

    return validatePredictionResponse(data);
}

async function getBackendPrediction(file) {
    const url =
        window.APP_CONFIG.API_BASE_URL +
        window.APP_CONFIG.PREDICT_ENDPOINT;

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(url, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        let detail = "Prediction request failed.";

        try {
            const errorData = await response.json();

            if (errorData.detail) {
                detail = errorData.detail;
            }
        } catch (_) {
            // Ignore JSON parsing errors.
        }

        throw new Error(detail);
    }

    const data = await response.json();

    return validatePredictionResponse(data);
}

async function predictXray(file) {
    if (window.APP_CONFIG.MOCK_MODE) {
        return getMockPrediction();
    }

    return getBackendPrediction(file);
}

async function requestGradcam(file, pathology) {
    if (window.APP_CONFIG.MOCK_MODE) {
        throw new Error(
            "Grad-CAM is not connected while the frontend is running in mock mode."
        );
    }

    const url =
        window.APP_CONFIG.API_BASE_URL +
        window.APP_CONFIG.GRADCAM_ENDPOINT;

    const formData = new FormData();

    formData.append("file", file);
    formData.append("pathology", pathology);

    const response = await fetch(url, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        throw new Error(
            "The Grad-CAM service could not generate an explanation."
        );
    }

    return response.blob();
}

window.TeamCoreAPI = {
    predictXray,
    requestGradcam
};
