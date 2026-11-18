# Retina Screen: Diabetic Retinopathy Detector

Classifies diabetic retinopathy (grades 0-4) from retinal fundus images, with a Grad-CAM heatmap
and a low-confidence "manual review" flag.

```
dr-detector/
  frontend/    React + Vite app  -> Render Static Site
  backend/     FastAPI + model   -> Hugging Face Space (Docker)
  notebooks/   Kaggle training notebooks
```

## Frontend (local)
```
cd frontend
cp .env.example .env     # set VITE_API_URL
npm install
npm run dev
```

## Deploy frontend on Render (Static Site)
- Root Directory: `frontend`
- Build Command: `npm install && npm run build`
- Publish Directory: `dist`
- Environment variable: `VITE_API_URL` = your backend URL (no trailing slash)
- After changing the variable: Manual Deploy -> Clear build cache & deploy

## API contract
See `frontend/src/api.js` (top comment). The backend must implement `GET /health` and `POST /predict`.
