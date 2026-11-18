import { useEffect, useRef, useState } from "react";
import { API_URL, GRADES, checkHealth, predict } from "./api.js";

const MAX_MB = 10;
const toSrc = (b64) => (b64.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`);

function ServerStatus() {
  const [state, setState] = useState(API_URL ? "checking" : "unconfigured");
  useEffect(() => {
    if (!API_URL) return;
    checkHealth()
      .then((h) => setState(h.model_loaded ? "online" : "nomodel"))
      .catch(() => setState("offline"));
  }, []);
  const text = {
    unconfigured: "Analysis server not configured",
    checking: "Connecting to analysis server (first load can take up to a minute)",
    online: "Analysis server online",
    nomodel: "Server is up but the model is not loaded",
    offline: "Analysis server unreachable",
  }[state];
  return <p className={`status s-${state}`} role="status"><span className="dot" />{text}</p>;
}

function CompareViewer({ original, heatmap }) {
  const [pos, setPos] = useState(50);
  return (
    <figure className="viewer">
      <div className="frame">
        <img src={original} alt="Uploaded retinal fundus" />
        <img src={heatmap} alt="Grad-CAM heatmap of regions that influenced the prediction"
             className="top" style={{ clipPath: `inset(0 0 0 ${pos}%)` }} />
        <div className="handle" style={{ left: `${pos}%` }} aria-hidden="true" />
      </div>
      <input type="range" min="0" max="100" value={pos}
             onChange={(e) => setPos(+e.target.value)}
             aria-label="Slide between original image and heatmap" />
      <figcaption>Left: your image. Right: where the model looked. Warm areas influenced the grade most.</figcaption>
    </figure>
  );
}

function Result({ r, original }) {
  const gradeLabel = r.label || GRADES[r.grade];
  return (
    <section className="result" aria-live="polite">
      <div className="grid">
        <div>
          <h2>Result</h2>
          <p className={`grade g${r.grade}`}>{gradeLabel}</p>
          <p className="meta">Grade {r.grade} of 4, confidence {(r.confidence * 100).toFixed(1)}%</p>
          {r.low_confidence && (
            <div className="warn" role="alert">
              <strong>Low confidence. Manual review recommended.</strong>
              <span>The model is not sure about this image
                {r.threshold ? ` (below the ${(r.threshold * 100).toFixed(0)}% threshold)` : ""}.
                Have a clinician examine it.</span>
            </div>
          )}
          <h3>Probability by grade</h3>
          <ul className="bars">
            {GRADES.map((g, i) => (
              <li key={g} className={i === r.grade ? "top" : ""}>
                <span>{g}</span>
                <span className="track"><span className={`fill g${i}`} style={{ width: `${r.probabilities[i] * 100}%` }} /></span>
                <span className="pct">{(r.probabilities[i] * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
        </div>
        {r.gradcam_base64
          ? <CompareViewer original={original} heatmap={toSrc(r.gradcam_base64)} />
          : <figure className="viewer"><div className="frame"><img src={original} alt="Uploaded retinal fundus" /></div></figure>}
      </div>
    </section>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [drag, setDrag] = useState(false);
  const input = useRef(null);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  function choose(f) {
    setResult(null); setError("");
    if (!f) return;
    if (!f.type.startsWith("image/")) return setError("Please choose an image file (JPG or PNG).");
    if (f.size > MAX_MB * 1024 * 1024) return setError(`Image is larger than ${MAX_MB} MB. Choose a smaller file.`);
    setFile(f); setPreview(URL.createObjectURL(f));
  }

  async function analyse() {
    setBusy(true); setError(""); setResult(null);
    try { setResult(await predict(file)); }
    catch (e) {
      setError(e.message === "NOT_CONFIGURED"
        ? "The analysis server is not configured. Set VITE_API_URL and rebuild."
        : e.message || "Could not analyse the image.");
    } finally { setBusy(false); }
  }

  function reset() { setFile(null); setPreview(null); setResult(null); setError(""); if (input.current) input.current.value = ""; }

  return (
    <>
      <header className="top">
        <div className="wrap">
          <h1>Retina Screen</h1>
          <p className="sub">Diabetic retinopathy grading from a fundus photograph, with the reasoning shown.</p>
          <ServerStatus />
        </div>
      </header>

      <main className="wrap">
        <section className="upload">
          {!preview ? (
            <label className={`drop ${drag ? "over" : ""}`}
                   onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                   onDragLeave={() => setDrag(false)}
                   onDrop={(e) => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files[0]); }}>
              <input ref={input} type="file" accept="image/*" onChange={(e) => choose(e.target.files[0])} />
              <strong>Choose a retinal fundus image</strong>
              <span>or drop it here. JPG or PNG, up to {MAX_MB} MB.</span>
            </label>
          ) : (
            <div className="picked">
              <img src={preview} alt="Selected retinal fundus preview" />
              <div className="actions">
                <p>{file.name}</p>
                <button className="primary" onClick={analyse} disabled={busy}>{busy ? "Analysing…" : "Analyse image"}</button>
                <button className="ghost" onClick={reset} disabled={busy}>Choose another image</button>
              </div>
            </div>
          )}
          {busy && <p className="hint">Running the model and building the heatmap. If the server was idle this can take up to a minute.</p>}
          {error && <p className="error" role="alert">{error}</p>}
        </section>

        {result && <Result r={result} original={preview} />}

        <section className="about">
          <h2>How it works</h2>
          <p>The image is cleaned (borders cropped, contrast enhanced) and passed to a convolutional neural network trained on labelled fundus photographs. It returns one of five grades, from No DR to Proliferative DR.</p>
          <p>A Grad-CAM heatmap highlights the regions that drove the decision, so a reviewer can check that the model looked at lesions and not at artefacts. When the model is unsure, the result is flagged for manual review instead of presenting a guess as fact.</p>
        </section>
      </main>

      <footer className="wrap disclaimer">
        This is an educational project, not a medical device. It does not provide a diagnosis. Consult a qualified eye specialist for any medical decision.
      </footer>
    </>
  );
}
