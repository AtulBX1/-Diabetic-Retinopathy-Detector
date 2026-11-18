// All backend communication lives here. Contract (backend must follow it):
//
// GET  /health  -> { "status": "ok", "model_loaded": true }
// POST /predict (multipart/form-data, field "file") ->
// {
//   "grade": 0-4,
//   "label": "Moderate",
//   "confidence": 0.0-1.0,
//   "probabilities": [p0, p1, p2, p3, p4],
//   "low_confidence": true|false,
//   "threshold": 0.6,
//   "gradcam_base64": "<png base64, with or without data: prefix>"
// }

const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export const API_URL = BASE;
export const GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];

async function request(path, options = {}, timeoutMs = 90000) {
  if (!BASE) throw new Error("NOT_CONFIGURED");
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(BASE + path, { ...options, signal: ctrl.signal });
    if (!res.ok) {
      let detail = "";
      try { detail = (await res.json()).detail; } catch { /* ignore */ }
      throw new Error(detail || `Server returned ${res.status}`);
    }
    return await res.json();
  } catch (e) {
    if (e.name === "AbortError") throw new Error("The server took too long to respond. It may be waking up; try again in a minute.");
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export const checkHealth = () => request("/health", {}, 60000);

export function predict(file) {
  const body = new FormData();
  body.append("file", file);
  return request("/predict", { method: "POST", body });
}
