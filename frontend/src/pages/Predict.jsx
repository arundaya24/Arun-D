import { useEffect, useRef, useState } from "react";
import { api, labelOf, pct } from "../api.js";

export default function Predict() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!file) return setPreview(null);
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function pick(f) {
    if (!f) return;
    setResult(null);
    setError("");
    setFile(f);
  }

  async function submit() {
    setLoading(true);
    setError("");
    try {
      setResult(await api.predict(file));
    } catch (e) {
      setResult(null);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="predict">
      <div
        className={`drop ${dragging ? "over" : ""} ${preview ? "filled" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]); }}
      >
        {preview ? (
          <img src={preview} alt="Gambar yang akan diklasifikasi" />
        ) : (
          <div className="drop-hint">
            <strong>Seret gambar ke sini</strong>
            <span>atau pilih dari komputer. JPG, PNG, WEBP, maks. 5 MB.</span>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          hidden
          onChange={(e) => pick(e.target.files[0])}
        />
        <div className="drop-actions">
          <button className="ghost" onClick={() => inputRef.current.click()}>
            {file ? "Ganti gambar" : "Pilih gambar"}
          </button>
          <button className="primary" disabled={!file || loading} onClick={submit}>
            {loading ? "Menganalisis…" : "Klasifikasikan"}
          </button>
        </div>
      </div>

      <div className="verdict" aria-live="polite">
        {error && <p className="error">{error}</p>}
        {!error && !result && (
          <p className="muted">
            Hasil prediksi dan tingkat keyakinan model akan muncul di sini. Model dilatih pada 10
            kelas: pesawat, mobil, burung, kucing, rusa, anjing, katak, kuda, kapal, dan truk.
          </p>
        )}
        {result && (
          <>
            <p className="kicker">Model memperkirakan</p>
            <h2 className="answer">{labelOf(result.predicted)}</h2>
            <p className="confidence">
              Keyakinan <b>{pct(result.confidence)}</b>
              {result.confidence < 0.5 && <span className="warn"> · rendah, gambar mungkin di luar 10 kelas</span>}
            </p>
            <ul className="bars">
              {result.top_k.map((t) => (
                <li key={t.label}>
                  <span>{labelOf(t.label)}</span>
                  <div className="track"><div className="fill" style={{ width: pct(t.score) }} /></div>
                  <em>{pct(t.score)}</em>
                </li>
              ))}
            </ul>
            <p className="meta">
              Diproses {result.inference_ms} ms dengan model {result.model_version}
            </p>
          </>
        )}
      </div>
    </section>
  );
}
