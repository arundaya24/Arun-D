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
  if (conf >= 0.8) return "#059669";
  if (conf >= 0.6) return "#d97706";
  return "#e11d48";
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
          <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.03em", color: "var(--text-tertiary)", textTransform: "uppercase" }}>
            Terdeteksi sebagai
          </div>
          <div style={{ fontSize: 17, fontWeight: 700, letterSpacing: "-0.02em", color: "var(--text-primary)", marginTop: 3 }}>
            {labelOf(predicted)}
          </div>
        </div>

        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
          {is_human && <span className="pill-human-tag">Manusia</span>}
          <span className="pill-engine-tag">{engine_used || "Dual-Engine"}</span>
        </div>
      </div>

      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 12, fontWeight: 500 }}>
          <span style={{ color: "var(--text-tertiary)" }}>Tingkat keyakinan</span>
          <span style={{ color, fontWeight: 700 }}>{pct(confidence)}</span>
        </div>
        <div style={{ height: 6, borderRadius: 999, background: "var(--bg-subtle)", overflow: "hidden" }}>
          <div
            style={{
              width: `${widthPct}%`,
              height: "100%",
              background: color,
              borderRadius: 999,
              transition: "width 0.5s var(--ease-smooth)",
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
            padding: "10px 14px",
            borderRadius: 12,
            border: "1px solid var(--accent-amber-border)",
            background: "var(--accent-amber-tint)",
            color: "var(--accent-amber)",
            fontSize: 12,
            fontWeight: 500,
          }}
        >
          <IconAlert />
          <span>Keyakinan model di bawah 60%, tolong verifikasi hasilnya ya.</span>
        </div>
      )}

      {top_k.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-tertiary)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
            Kandidat teratas
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {top_k.slice(0, 3).map((item, idx) => (
              <div key={idx} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13 }}>
                <span style={{ color: idx === 0 ? "var(--text-primary)" : "var(--text-secondary)", fontWeight: idx === 0 ? 600 : 400, maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {labelOf(item.label)}
                </span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: idx === 0 ? "var(--accent-primary)" : "var(--text-tertiary)", fontWeight: 600 }}>
                  {pct(item.score)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid var(--border-hairline)", paddingTop: 12, marginTop: 2 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)" }}>
          {inference_ms || 0} ms
        </span>

        {!fs.submitted ? (
          <div style={{ display: "flex", gap: 8 }}>
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
          <span style={{ fontSize: 12, fontWeight: 600, color: "var(--status-success)" }}>
            Tercatat, terima kasih!
          </span>
        )}
      </div>

      {fs.showing && !fs.submitted && (
        <div style={{ display: "flex", gap: 8, marginTop: 4, paddingTop: 12, borderTop: "1px solid var(--border-hairline)" }}>
          <input
            type="text"
            placeholder="Tulis label yang benar..."
            value={fs.labelInput}
            onChange={(e) => onFeedback(prediction_id, predicted, e.target.value, false, true, true)}
            style={{
              flex: 1,
              background: "var(--bg-elevated)",
              border: "1px solid transparent",
              borderRadius: 12,
              padding: "8px 14px",
              color: "var(--text-primary)",
              fontFamily: "var(--font-sans)",
              fontSize: 13,
              outline: "none",
              transition: "all 0.2s var(--ease-smooth)",
            }}
            onFocus={(e) => { e.target.style.borderColor = "var(--accent-primary)"; e.target.style.boxShadow = "0 0 0 3px rgba(79,70,229,0.1)"; }}
            onBlur={(e) => { e.target.style.borderColor = "transparent"; e.target.style.boxShadow = "none"; }}
          />
          <button
            type="button"
            className="btn-shimmer-cta"
            disabled={!fs.labelInput.trim()}
            onClick={() => onFeedback(prediction_id, predicted, fs.labelInput.trim(), false)}
            style={{ padding: "8px 16px", fontSize: 12, flex: "none" }}
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
        text: "Halo! Kirim foto dan aku akan coba tebak objek atau orang yang ada di dalamnya. Kalau hasilnya kurang tepat, kamu bisa langsung koreksi.",
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
      text: userText || (currentFile ? `Foto dikirim: ${currentFile.name}` : ""),
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
      {/* ── Left Column: Chat ── */}
      <div className="bento-card chat-card">
        <div className="chat-header">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="card-status-dot" />
            <span style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>Tanya AI</span>
          </div>
          <span style={{ fontSize: 12, fontWeight: 500, color: "var(--status-success)", background: "var(--status-success-bg)", border: "1px solid var(--status-success-border)", padding: "3px 10px", borderRadius: "var(--radius-full)" }}>
            Online
          </span>
        </div>

        <div className="chat-scroll-area">
          {messages.map((msg) => (
            <div key={msg.id} className={`chat-bubble-row ${msg.role}`}>
              <div className="chat-badge-avatar">
                {msg.role === "user" ? "Kamu" : "AI"}
              </div>

              <div className="chat-bubble-content">
                {msg.imageUrl && (
                  <img src={msg.imageUrl} alt="Foto yang dikirim" className="chat-image-preview" />
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
              <div className="chat-badge-avatar">AI</div>
              <div className="chat-bubble-content" style={{ padding: "14px 20px" }}>
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
              title="Lampirkan foto"
            >
              <IconCamera />
            </button>

            <textarea
              className="chat-textarea"
              rows={1}
              placeholder="Tanya sesuatu, atau kirim foto..."
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
              style={{ flex: "none", width: 44, height: 44, padding: 0 }}
              title="Kirim"
            >
              <IconSend />
            </button>
          </div>
        </div>
      </div>

      {/* ── Right Column: Sidebar ── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <div className="bento-card" style={{ padding: 24, gap: 16 }}>
          <div className="card-header">
            <span style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>Info Model</span>
            <span style={{ fontSize: 11, fontWeight: 600, color: "var(--status-success)", background: "var(--status-success-bg)", border: "1px solid var(--status-success-border)", padding: "2px 8px", borderRadius: "var(--radius-full)" }}>Aktif</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 14, marginTop: 4 }}>
            {[
              { label: "Engine Utama", val: "MobileNetV3" },
              { label: "Fallback", val: "Arun-D CNN" },
              { label: "Deteksi Manusia", val: "Aktif" },
              { label: "Jumlah Kategori", val: "1.000+ kelas" },
            ].map(({ label, val }) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13 }}>
                <span style={{ color: "var(--text-tertiary)", fontWeight: 500 }}>{label}</span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>{val}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bento-card" style={{ padding: 24, gap: 16 }}>
          <div className="card-header">
            <span style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>Latih Ulang Model</span>
          </div>

          <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65 }}>
            Setiap koreksi yang kamu berikan akan disimpan dan dipakai untuk melatih ulang model supaya makin pintar.
          </p>

          <div style={{ marginTop: 4 }}>
            <button
              type="button"
              className="btn-secondary"
              onClick={handleRetrainTrigger}
              disabled={retrainStatus.status === "running"}
              style={{ width: "100%", fontSize: 13, fontWeight: 600 }}
            >
              {retrainStatus.status === "running" ? "Sedang memproses..." : "Jalankan Fine-Tuning"}
            </button>
            {retrainStatus.message && (
              <div style={{ fontSize: 12, color: "var(--text-tertiary)", marginTop: 10, lineHeight: 1.55 }}>
                {retrainStatus.message}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
