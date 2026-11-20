# Retina Screen: Diabetic Retinopathy Detector

Classifies diabetic retinopathy (grades 0-4) from retinal fundus images and returns
a Grad-CAM visualization and a low-confidence flag.

```text
frontend/    React + Vite web app
backend/     FastAPI API and model
notebooks/   Training notebooks
scripts/     API validation utilities
```

## Run the backend locally

Python 3.13 is supported by the pinned backend dependencies. The trained model weights
must be available at `backend/model/dr_model.keras` before predictions can be made.
The model configuration is in `backend/model/config.json`.

In Windows PowerShell, from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On macOS or Linux, activate the virtual environment with
`source .venv/bin/activate` instead of the PowerShell activation command.

The API is available at `http://127.0.0.1:8000`; interactive API documentation is at
`http://127.0.0.1:8000/docs`. Check `GET /health` and confirm `model_loaded` is `true`.
If the model weights are missing or fail to load, health reports `model_loaded: false`
and prediction requests cannot succeed.

`POST /predict` accepts one multipart `file` field containing a JPEG or PNG image. The
file must be no larger than 10 MB, and the image must be at least 64 pixels on each side.
The response contains the predicted grade, label, confidence, class probabilities,
low-confidence status, threshold, and a base64-encoded Grad-CAM PNG.

Set `ALLOWED_ORIGINS` to a comma-separated list of frontend origins if the frontend is
served from somewhere other than `http://localhost:5173`.

## Run backend tests and check real images

With the virtual environment activated, run from the repository root:

```powershell
python -m pip install -r backend/requirements-dev.txt
python -m pytest backend/tests
```

To check a folder of real PNG images against a running API, name files
`grade_<n>_something.png` (where `<n>` is the true grade from 0 to 4), then run:

```powershell
python scripts/check_api.py path\to\images --url http://127.0.0.1:8000
```

The script prints each true grade, predicted grade, confidence, low-confidence status,
and request time, followed by overall accuracy.

## Run the frontend locally

In a separate terminal:

```powershell
cd frontend
Copy-Item .env.example .env
npm install
npm run dev
```

Set `VITE_API_URL` in `frontend/.env` to the backend base URL, without a trailing slash.

## Deploy the frontend on Render

- Root Directory: `frontend`
- Build Command: `npm install && npm run build`
- Publish Directory: `dist`
- Environment variable: `VITE_API_URL` set to the deployed backend base URL (no trailing slash)
- After changing the variable, manually deploy and clear the build cache.
