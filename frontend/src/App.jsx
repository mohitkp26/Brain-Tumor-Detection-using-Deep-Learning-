import React, { useCallback, useRef, useState } from "react";

const CLASS_LABELS = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  no_tumor: "No Tumor Detected",
  pituitary: "Pituitary Tumor",
};

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | scanning | done | error
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const inputRef = useRef(null);

  const handleFile = useCallback((selected) => {
    if (!selected) return;
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setResult(null);
    setStatus("idle");
    setErrorMsg("");
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files?.[0]);
  };

  const runAnalysis = async () => {
    if (!file) return;
    setStatus("scanning");
    setErrorMsg("");

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Prediction failed.");
      }

      // small delay so the scan animation is visible even on fast responses
      setTimeout(() => {
        setResult(data);
        setStatus("done");
      }, 1200);
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setStatus("idle");
    setErrorMsg("");
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">NEUROSCAN</span>
        </div>
        <span className="topbar-tag">MRI TUMOR CLASSIFICATION · CNN INFERENCE</span>
      </header>

      <main className="layout">
        <section className="intro">
          <p className="eyebrow">01 — UPLOAD</p>
          <h1>
            Read the scan
            <br />
            before the radiologist does.
          </h1>
          <p className="intro-copy">
            Drop an MRI slice below. A convolutional network trained on
            labeled glioma, meningioma, and pituitary cases returns a
            classification with per-class confidence in seconds.
          </p>
          <ul className="legend">
            <li><span className="dot dot-glioma" />Glioma</li>
            <li><span className="dot dot-meningioma" />Meningioma</li>
            <li><span className="dot dot-pituitary" />Pituitary</li>
            <li><span className="dot dot-none" />No tumor</li>
          </ul>
        </section>

        <section className="scan-panel">
          <div
            className={`dropzone ${previewUrl ? "has-image" : ""}`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/png, image/jpeg"
              hidden
              onChange={(e) => handleFile(e.target.files?.[0])}
            />

            {!previewUrl && (
              <div className="dropzone-placeholder">
                <span className="dropzone-icon">+</span>
                <p>Drop MRI image or click to browse</p>
                <p className="dropzone-hint">PNG or JPEG</p>
              </div>
            )}

            {previewUrl && (
              <div className="image-wrap">
                <img src={previewUrl} alt="Uploaded MRI scan" />
                {status === "scanning" && <div className="sweep" aria-hidden="true" />}
                {status === "scanning" && (
                  <div className="grid-overlay" aria-hidden="true" />
                )}
              </div>
            )}
          </div>

          <div className="controls">
            <button
              className="btn-primary"
              onClick={runAnalysis}
              disabled={!file || status === "scanning"}
            >
              {status === "scanning" ? "Analyzing…" : "Run Analysis"}
            </button>
            <button className="btn-ghost" onClick={reset} disabled={!file}>
              Clear
            </button>
          </div>

          {status === "error" && (
            <p className="error-box">{errorMsg}</p>
          )}

          {status === "done" && result && (
            <div className="result-box">
              <p className="eyebrow">02 — RESULT</p>
              <div className="result-headline">
                <span
                  className={`result-class result-${result.predicted_class}`}
                >
                  {CLASS_LABELS[result.predicted_class] || result.predicted_class}
                </span>
                <span className="result-confidence">
                  {result.confidence}% confidence
                </span>
              </div>

              <div className="prob-bars">
                {Object.entries(result.probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, prob]) => (
                    <div className="prob-row" key={cls}>
                      <span className="prob-label">
                        {CLASS_LABELS[cls] || cls}
                      </span>
                      <div className="prob-track">
                        <div
                          className={`prob-fill prob-fill-${cls}`}
                          style={{ width: `${prob}%` }}
                        />
                      </div>
                      <span className="prob-value">{prob}%</span>
                    </div>
                  ))}
              </div>

              <p className="disclaimer">
                Model output only — not a clinical diagnosis. Confirm any
                finding with a licensed radiologist.
              </p>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        <span>Backend: Flask + TensorFlow/Keras CNN</span>
        <span>Frontend: React + Vite</span>
      </footer>
    </div>
  );
}
