import { useEffect, useRef, useState } from "react";
import { api, labelOf, pct } from "../api.js";

/* ─────────────────────────── SVGs (Zero Emojis) ─────────────────────────── */

function IconCamera() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  );
}

function IconSend() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function IconClose() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function IconAlert() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function IconCheck() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

let _id = 0;
const uid = () => `msg-${++_id}-${Date.now()}`;

function confidenceColor(conf) {
  if (conf >= 0.8) return "#4ade80";
  if (conf >= 0.6) return "#e59344";
  return "#f87171";
}

/* ─────────────────────────── Prediction Card ─────────────────────────── */

function PredictionCard({ prediction, onFeedback, feedbackState }) {
  const {
    predicted,
    confidence,
    top_k = [],
    is_low_confidence,
    is_human,
    engine_used,
    prediction_id,
    inference_ms,
  } = prediction;

  const fs = feedbackState || { showing: false, labelInput: "", submitted: false };
  const color = confidenceColor(confidence);
  const widthPct = Math.min(100, Math.round(confidence * 100));

  return (
    <div className="card-telemetry-panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: "0.06em", color: "var(--text-tertiary)", textTransform: "uppercase" }}>
            Hasil Klasifikasi
          </div>
          <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: "-0.025em", color: "var(--text-primary)", marginTop: 2 }}>
            {labelOf(predicted)}
          </div>
        </div>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
          {is_human && <span className="pill-human-tag">Subjek: Manusia</span>}
          <span className="pill-engine-tag">{engine_used || "Dual-Engine"}</span>
        </div>
      </div>

      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontFamily: "var(--font-mono)", fontSize: 11 }}>
          <span style={{ color: "var(--text-tertiary)" }}>Confidence Score</span>
          <span style={{ color, fontWeight: 600 }}>{pct(confidence)}</span>
        </div>
        <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.08)", overflow: "hidden" }}>
          <div
            style={{
              width: `${widthPct}%`,
              height: "100%",
              background: color,
              transition: "width 0.4s var(--ease-smooth)",
            }}
          />
        </div>
      </div>

      {is_low_confidence && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "8px 10px",
            borderRadius: 4,
            border: "1px solid rgba(229, 147, 68, 0.35)",
            background: "var(--accent-tint)",
            color: "var(--accent-amber)",
            fontSize: 11,
            fontFamily: "var(--font-mono)",
          }}
        >
          <IconAlert />
          <span>Status: Tingkat keyakinan rendah (&lt; 60%). Butuh verifikasi pengguna.</span>
        </div>
      )}

      {top_k.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-tertiary)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Distribusi Top Kategori
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {top_k.slice(0, 3).map((item, idx) => (
              <div key={idx} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 12 }}>
                <span style={{ color: "var(--text-secondary)", maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {labelOf(item.label)}
                </span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-tertiary)" }}>
                  {pct(item.score)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid var(--border-hairline)", paddingTop: 10, marginTop: 4 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-tertiary)" }}>
          Latensi: {inference_ms || 0} ms
        </span>

        {!fs.submitted ? (
          <div style={{ display: "flex", gap: 6 }}>
            <button
              type="button"
              className="btn-tactile btn-tactile-confirm"
              onClick={() => onFeedback(prediction_id, predicted, predicted, true)}
            >
              <IconCheck /> Benar
            </button>
            <button
              type="button"
              className="btn-tactile btn-tactile-correct"
              onClick={() => onFeedback(prediction_id, predicted, null, false, true)}
            >
              Koreksi
            </button>
          </div>
        ) : (
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "#4ade80" }}>
            Feedback Tercatat
          </span>
        )}
      </div>

      {fs.showing && !fs.submitted && (
        <div style={{ display: "flex", gap: 6, marginTop: 6, paddingTop: 8, borderTop: "1px solid var(--border-hairline)" }}>
          <input
            type="text"
            placeholder="Label yang benar..."
            value={fs.labelInput}
            onChange={(e) => onFeedback(prediction_id, predicted, e.target.value, false, true, true)}
            style={{
              flex: 1,
              background: "var(--bg-elevated)",
              border: "1px solid var(--border-hairline)",
              borderRadius: 4,
              padding: "6px 10px",
              color: "var(--text-primary)",
              fontFamily: "var(--font-sans)",
              fontSize: 12,
              outline: "none",
            }}
          />
          <button
            type="button"
            className="btn-shimmer-cta"
            disabled={!fs.labelInput.trim()}
            onClick={() => onFeedback(prediction_id, predicted, fs.labelInput.trim(), false)}
            style={{ padding: "6px 12px", fontSize: 11 }}
          >
            Simpan
          </button>
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────── Main Chat View ─────────────────────────── */

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [feedbackStates, setFeedbackStates] = useState({});
  const [statsData, setStatsData] = useState(null);
  const [retrainStatus, setRetrainStatus] = useState({ status: "idle", message: "" });

  const fileInputRef = useRef(null);
  const bottomRef = useRef(null);

  // Inisialisasi percakapan awal
  useEffect(() => {
    setMessages([
      {
        id: uid(),
        role: "assistant",
        text: "Sistem inferensi citra aktif. Lampirkan berkas citra untuk menjalankan klasifikasi objek dan deteksi manusia, atau ajukan pertanyaan.",
        prediction: null,
      },
    ]);
    loadFeedbackStats();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  async function loadFeedbackStats() {
    try {
      const s = await api.chatStats ? await api.chatStats() : null;
      if (s) setStatsData(s);
    } catch {
      // ignore
    }
  }

  function handleFileSelection(file) {
    if (!file) return;
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }

  function clearSelectedFile() {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function sendMessage() {
    if (isLoading) return;
    if (!inputText.trim() && !selectedFile) return;

    const userText = inputText.trim();
    const currentFile = selectedFile;
    const currentPreview = previewUrl;

    // Reset input state
    setInputText("");
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";

    // Tambahkan bubble user
    const userMsg = {
      id: uid(),
      role: "user",
      text: userText || (currentFile ? `Menganalisis berkas: ${currentFile.name}` : ""),
      imageUrl: currentPreview,
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    const assistantMsgId = uid();

    try {
      // Panggil backend API /api/chat
      const data = await api.chat(userText, currentFile);

      let prediction = null;
      if (data && data.predicted) {
        prediction = {
          prediction_id: data.prediction_id,
          predicted: data.predicted,
          confidence: data.confidence,
          top_k: data.top_k || [],
          engine_used: data.engine_used,
          is_low_confidence: data.is_low_confidence,
          is_human: data.is_human,
          inference_ms: data.inference_ms,
          model_version: data.model_version,
        };
      }

      const botText = data.chatbot_message || (prediction ? `Objek teridentifikasi sebagai ${labelOf(prediction.predicted)}.` : "Permintaan selesai.");

      setMessages((prev) => [
        ...prev,
        {
          id: assistantMsgId,
          role: "assistant",
          text: botText,
          prediction,
        },
      ]);
      loadFeedbackStats();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: assistantMsgId,
          role: "assistant",
          text: `Gagal memproses permintaan: ${err.message || "Server tidak merespons."}`,
          prediction: null,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleFeedback(predId, origLabel, labelInput, isCorrect, showInput = false, inputOnly = false) {
    if (showInput && !inputOnly) {
      setFeedbackStates((prev) => ({
        ...prev,
        [predId]: { showing: true, labelInput: prev[predId]?.labelInput || "", submitted: false },
      }));
      return;
    }

    if (inputOnly) {
      setFeedbackStates((prev) => ({
        ...prev,
        [predId]: { ...prev[predId], labelInput: labelInput || "" },
      }));
      return;
    }

    try {
      await api.submitFeedback({
        prediction_id: predId,
        original_label: origLabel,
        correct_label: isCorrect ? origLabel : (labelInput || origLabel),
        is_correct: isCorrect,
      });

      setFeedbackStates((prev) => ({
        ...prev,
        [predId]: { showing: false, labelInput: "", submitted: true },
      }));
      loadFeedbackStats();
    } catch (e) {
      alert(`Gagal menyimpan feedback: ${e.message}`);
    }
  }

  async function handleRetrainTrigger() {
    try {
      setRetrainStatus({ status: "running", message: "Menginisialisasi retraining..." });
      const res = await api.retrainModel();
      setRetrainStatus({ status: "running", message: res.message || "Retraining sedang berjalan." });
      // Polling status singkat
      setTimeout(async () => {
        try {
          const s = await api.retrainStatus();
          setRetrainStatus(s);
        } catch {
          // ignore
        }
      }, 3000);
    } catch (e) {
      setRetrainStatus({ status: "error", message: e.message });
    }
  }

  return (
    <section className="chat-shell">
      {/* ── Left Column: Chat Terminal ── */}
      <div className="bento-card chat-card">
        <div className="chat-header">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="card-status-dot" />
            <span className="card-label">Terminal Inferensi Interaktif</span>
          </div>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-tertiary)" }}>
            PORT: 8000 / DUAL ENGINE
          </span>
        </div>

        <div className="chat-scroll-area">
          {messages.map((msg) => (
            <div key={msg.id} className={`chat-bubble-row ${msg.role}`}>
              <div className="chat-badge-avatar">
                {msg.role === "user" ? "USR" : "SYS"}
              </div>

              <div className="chat-bubble-content">
                {msg.imageUrl && (
                  <img src={msg.imageUrl} alt="Lampiran input" className="chat-image-preview" />
                )}
                <div>{msg.text}</div>

                {msg.prediction && (
                  <PredictionCard
                    prediction={msg.prediction}
                    feedbackState={feedbackStates[msg.prediction.prediction_id]}
                    onFeedback={handleFeedback}
                  />
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="chat-bubble-row assistant">
              <div className="chat-badge-avatar">SYS</div>
              <div className="chat-bubble-content" style={{ padding: "8px 14px" }}>
                <div className="pulse-loader">
                  <span className="pulse-dot" />
                  <span className="pulse-dot" />
                  <span className="pulse-dot" />
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input Bar */}
        <div className="chat-input-bar">
          {previewUrl && (
            <div className="chat-attachment-strip">
              <img src={previewUrl} alt="Thumbnail" className="chat-attachment-thumb" />
              <span className="chat-attachment-name">{selectedFile?.name || "lampiran_citra.jpg"}</span>
              <button
                type="button"
                className="chat-attachment-remove"
                onClick={clearSelectedFile}
                title="Hapus lampiran"
              >
                <IconClose />
              </button>
            </div>
          )}

          <div className="chat-input-row">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                if (e.target.files?.[0]) handleFileSelection(e.target.files[0]);
              }}
            />

            <button
              type="button"
              className="btn-icon-tactile"
              onClick={() => fileInputRef.current?.click()}
              title="Lampirkan berkas citra"
            >
              <IconCamera />
            </button>

            <textarea
              className="chat-textarea"
              rows={1}
              placeholder="Ketik instruksi atau lampirkan citra..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
            />

            <button
              type="button"
              className="btn-shimmer-cta"
              disabled={isLoading || (!inputText.trim() && !selectedFile)}
              onClick={sendMessage}
              style={{ flex: "none", width: 42, height: 42, padding: 0 }}
              title="Kirim pesan"
            >
              <IconSend />
            </button>
          </div>
        </div>
      </div>

      {/* ── Right Column: Telemetry & Active Learning Sidebar ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div className="bento-card" style={{ padding: 20 }}>
          <div className="card-header">
            <span className="card-label">Konfigurasi Model</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "#4ade80" }}>ONLINE</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 4 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
              <span style={{ color: "var(--text-tertiary)" }}>Primary Engine</span>
              <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>MobileNetV3</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
              <span style={{ color: "var(--text-tertiary)" }}>Fallback Engine</span>
              <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>Arun-D CNN</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
              <span style={{ color: "var(--text-tertiary)" }}>Deteksi Manusia</span>
              <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent-amber)" }}>TERPASANG</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
              <span style={{ color: "var(--text-tertiary)" }}>Total Kategori</span>
              <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>1.000+ Kelas</span>
            </div>
          </div>
        </div>

        <div className="bento-card" style={{ padding: 20 }}>
          <div className="card-header">
            <span className="card-label">Active Learning Buffer</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-tertiary)" }}>RETRAIN PIPELINE</span>
          </div>

          <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5, marginTop: 4 }}>
            Setiap koreksi yang Anda masukkan akan disimpan ke dataset lokal untuk melatih ulang head klasifikasi tanpa merusak bobot model utama.
          </p>

          <div style={{ marginTop: 12 }}>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleRetrainTrigger}
              disabled={retrainStatus.status === "running"}
              style={{ width: "100%", fontSize: 11, fontFamily: "var(--font-mono)", letterSpacing: "0.04em" }}
            >
              {retrainStatus.status === "running" ? "MEMPROSES RETRAINING..." : "JALANKAN FINE-TUNING"}
            </button>
            {retrainStatus.message && (
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--text-tertiary)", marginTop: 8, lineHeight: 1.4 }}>
                {retrainStatus.message}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
