import { useEffect, useRef, useState } from "react";
import { api, labelOf, pct } from "../api.js";

export default function Predict() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [inferenceResult, setInferenceResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  function handleFileSelection(file) {
    if (!file) return;
    setInferenceResult(null);
    setErrorMessage("");
    setSelectedFile(file);
  }

  async function executeInference() {
    if (!selectedFile || isProcessing) return;
    setIsProcessing(true);
    setErrorMessage("");
    try {
      const response = await api.predict(selectedFile);
      setInferenceResult(response);
    } catch (err) {
      setInferenceResult(null);
      setErrorMessage(err.message);
    } finally {
      setIsProcessing(false);
    }
  }

  return (
    <section className="bento-grid">
      <div className="bento-card">
        <div className="card-header">
          <span style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>Upload Foto</span>
          <span className="card-status-dot" />
        </div>

        <div
          className={`drop-viewport ${isDragOver ? "dragging" : ""} ${previewUrl ? "has-preview" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragOver(false);
            if (e.dataTransfer.files?.[0]) {
              handleFileSelection(e.dataTransfer.files[0]);
            }
          }}
          onClick={() => {
            if (!previewUrl) fileInputRef.current?.click();
          }}
        >
          {previewUrl ? (
            <img src={previewUrl} alt="Preview input" className="drop-image" />
          ) : (
            <div className="drop-empty-prompt">
              <div className="drop-icon-box">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <strong className="drop-heading">Seret foto ke sini atau klik untuk memilih</strong>
              <span className="drop-meta">Format JPG, PNG, WEBP hingga 5 MB</span>
            </div>
          )}
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          hidden
          onChange={(e) => {
            if (e.target.files?.[0]) {
              handleFileSelection(e.target.files[0]);
            }
          }}
        />

        <div className="action-row">
          <button
            type="button"
            className="btn-secondary"
            onClick={() => fileInputRef.current?.click()}
          >
            {selectedFile ? "Ganti Foto" : "Pilih Foto"}
          </button>
          <button
            type="button"
            className="btn-shimmer-cta"
            disabled={!selectedFile || isProcessing}
            onClick={executeInference}
          >
            {isProcessing ? "Menganalisis..." : "Kenali Gambar"}
          </button>
        </div>
      </div>

      <div className="bento-card" aria-live="polite">
        <div className="card-header">
          <span className="card-label">Hasil Deteksi</span>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Top prediksi</span>
        </div>

        {errorMessage && <div className="error-banner">{errorMessage}</div>}

        {!errorMessage && !inferenceResult && (
          <div className="empty-state">
            <h3 className="empty-state-title">Belum ada hasil</h3>
            <p className="empty-state-text">
              Upload foto dulu di sebelah kiri, lalu klik "Kenali Gambar" untuk melihat hasil deteksinya.
            </p>
          </div>
        )}

        {inferenceResult && (
          <div className="verdict-bento">
            <div className="verdict-header-block">
              <span className="verdict-kicker">Terdeteksi sebagai</span>
              <h2 className="verdict-title">{labelOf(inferenceResult.predicted)}</h2>
            </div>

            <div className="metrics-strip">
              <div className="metric-item">
                <span className="metric-label">Keyakinan</span>
                <span className="metric-val">{pct(inferenceResult.confidence)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Waktu proses</span>
                <span className="metric-val">{inferenceResult.inference_ms} ms</span>
              </div>
            </div>

            <ul className="distribution-list">
              {inferenceResult.top_k.map((item, index) => (
                <li key={item.label} className="distribution-row">
                  <span className="distribution-name">{labelOf(item.label)}</span>
                  <div className="distribution-track">
                    <div
                      className={`distribution-fill ${index === 0 ? "lead" : ""}`}
                      style={{ width: pct(item.score) }}
                    />
                  </div>
                  <span className="distribution-val">{pct(item.score)}</span>
                </li>
              ))}
            </ul>

            <div className="telemetry-row">
              <span>Model: {inferenceResult.model_version}</span>
              <span style={{ color: "var(--status-success)", fontWeight: 600 }}>Selesai</span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
