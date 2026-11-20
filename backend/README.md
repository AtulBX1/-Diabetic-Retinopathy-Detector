# Backend

Run the API from the repository root:

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Place `dr_model.keras` and `config.json` in `backend/model/`.

## Environment

Set `ALLOWED_ORIGINS` to a comma-separated list of frontend origins. The default is
`http://localhost:5173`.

## API

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Accept: application/json" \
  -F "file=@/path/to/fundus.jpg"
```

Only JPEG and PNG files no larger than 10 MB are accepted. Images must be at least
64 pixels on each side.

## Tests and API check

Install the development dependencies and run the backend tests from the repository root:

```bash
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend/tests
```

To compare predictions against real PNG images named `grade_<n>_something.png`:

```bash
python scripts/check_api.py /path/to/images --url http://127.0.0.1:8000
```

The checker prints each image's true and predicted grades, confidence, low-confidence
status, request time, and the overall accuracy.

## Assumptions

1. The notebook is unavailable in this checkout, so the standard APTOS preprocessing
   and Grad-CAM behavior described by the project context is implemented.
2. The model input uses float32 RGB values in the 0-255 range.
3. `crop_resize(image, 224)` is the base image for prediction and Grad-CAM.
4. Configured class names correspond positionally to output indexes 0 through 4.
5. If `ALLOWED_ORIGINS=*`, CORS credentials are disabled to remain browser-safe.
