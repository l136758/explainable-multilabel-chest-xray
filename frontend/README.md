# Frontend — Team Core

This folder contains the web interface for Team Core's Explainable Multi-Label Chest X-Ray prototype.

## Current status

- Frontend technology: **HTML, CSS, and vanilla JavaScript**
- Current mode: **Mock mode**
- Backend integration: **Pending**
- Grad-CAM integration: **Pending**
- Report export: **Prototype summary only; final PDF export is pending**
- Clinical positioning: **Research and educational prototype only**

The frontend intentionally does **not** load the PyTorch model or repeat model preprocessing. Model inference, preprocessing, thresholds, and Grad-CAM generation belong to the backend / XAI pipeline.

## Run locally

From the `frontend` folder:

```bash
py -m http.server 5500
```

or:

```bash
python -m http.server 5500
```

Then open:

```text
http://127.0.0.1:5500/
```

## Main files

```text
frontend/
├── index.html
├── analyze.html
├── css/
│   └── styles.css
├── js/
│   ├── config.js
│   ├── api.js
│   └── analyze.js
├── mocks/
│   └── prediction.json
└── assets/
```

## Mock mode

Mock mode is controlled in:

```text
frontend/js/config.js
```

Current setting:

```javascript
MOCK_MODE: true
```

Keep it `true` until the backend response contract is confirmed and the FastAPI service is ready.

## Expected model behavior

The classifier returns 14 pathology probabilities. The frontend:

1. displays all 14 model scores,
2. highlights pathologies that cross their saved class-specific threshold,
3. sorts findings from highest to lowest model score,
4. derives a `No Finding` state only when no pathology crosses its threshold,
5. displays a class-specific Grad-CAM explanation when available.

`No Finding` is **not** a 15th model output.

## Expected label order

The frontend validates the exact model label order:

1. Atelectasis
2. Consolidation
3. Infiltration
4. Pneumothorax
5. Edema
6. Emphysema
7. Fibrosis
8. Effusion
9. Pneumonia
10. Pleural_Thickening
11. Cardiomegaly
12. Nodule
13. Mass
14. Hernia

## File upload

The UI currently allows:

- PNG
- JPG / JPEG

PDF uploads are not supported because model inference requires an image file.

## Integration contract

See:

```text
docs/frontend_backend_handoff.md
```

for the proposed request/response contract and open integration decisions.
