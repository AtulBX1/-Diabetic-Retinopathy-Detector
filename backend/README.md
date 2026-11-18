# Backend (to be built)

Planned layout:
```
backend/
  app/main.py        FastAPI: /health, /predict
  app/preprocess.py  same preprocessing as training
  app/gradcam.py     Grad-CAM overlay
  model/             put dr_model.keras here (git-ignored)
  requirements.txt
  Dockerfile         for Hugging Face Spaces
```
Contract: see `frontend/src/api.js`.
