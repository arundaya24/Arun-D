import { useEffect, useRef, useState } from "react";
import { api, labelOf, pct } from "../api.js";

/* ─────────────────────────── helpers ─────────────────────────── */

let _id = 0;
const uid = () => `msg-${++_id}-${Date.now()}`;

function confidenceColor(conf) {
  if (conf >= 0.8) return "#22c55e";   // green
  if (conf >= 0.6) return "#eab308";   // yellow
  return "#ef4444";                     // red
}

/* ─────────────────────────── sub-components ──────────────────── */

function TypingIndicator() {
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: "0.5rem" }}>
      {/* avatar */}
      <div
        style={{
          width: 28,
          height: 28,
          borderRadius: "50%",
          background: "var(--accent)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 14,
          flexShrink: 0,
        }}
      >
        🤖
      </div>
      <div
        style={{
          background: "var(--card-bg)",
          border: "1px solid var(--border)",
          borderRadius: "0 12px 12px 12px",
          padding: "0.6rem 1rem",
          display: "flex",
          gap: 4,
          alignItems: "center",
        }}
      >
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            style={{
              display: "inline-block",
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: "var(--text-primary)",
              opacity: 0.6,
              animation: `typingBounce 1.2s ease-in-out ${i * 0.2}s infinite`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

function UserBubble({ message }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: "0.3rem",
      }}
    >
      {message.imageUrl && (
        <img
          src={message.imageUrl}
          alt="uploaded"
          style={{
            maxWidth: 180,
            maxHeight: 180,
            borderRadius: 8,
            objectFit: "cover",
            border: "1px solid var(--border)",
          }}
        />
      )}
      {message.text && (
        <div
          style={{
            background: "var(--accent)",
            color: "#fff",
            borderRadius: "12px 12px 0 12px",
            padding: "0.55rem 0.9rem",
            maxWidth: "75%",
            wordBreak: "break-word",
            lineHeight: 1.5,
            fontSize: "0.9rem",
          }}
        >
          {message.text}
        </div>
      )}
    </div>
  );
}

function PredictionCard({ prediction, messageId, feedbackState, onFeedback }) {
  const { predicted, confidence, top3 = [], is_low_confidence, engine_used, prediction_id } = prediction;
  const original_label = predicted;
  const fs = feedbackState || { showing: false, labelInput: "", submitted: false };
  const color = confidenceColor(confidence);
  const pctVal = pct(confidence);         // e.g. "87.3%"
  const widthPct = Math.min(100, Math.round(confidence * 100));

  const engineLabel =
    engine_used === "mobilenet" ? "MobileNetV3"
    : engine_used === "arun_cnn" ? "ArunD CNN"
    : engine_used ?? "Unknown Engine";

  return (
    <div
      className="verdict-bento"
      style={{
        marginTop: "0.6rem",
        background: "rgba(255,255,255,0.03)",
        border: "1px solid var(--border)",
        borderRadius: 10,
        padding: "0.85rem 1rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.65rem",
        fontSize: "0.85rem",
      }}
    >
      {/* header */}
      <div className="verdict-header-block" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div className="verdict-kicker" style={{ fontSize: "0.7rem", opacity: 0.6, letterSpacing: 1, textTransform: "uppercase" }}>
            Hasil Prediksi
          </div>
          <div
            className="verdict-title"
            style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary)" }}
          >
            {labelOf(predicted)}
          </div>
        </div>
        <span
          style={{
            fontSize: "0.7rem",
            background: "rgba(255,255,255,0.08)",
            border: "1px solid var(--border)",
            borderRadius: 20,
            padding: "2px 10px",
            color: "var(--text-primary)",
            opacity: 0.8,
            flexShrink: 0,
          }}
        >
          {engineLabel}
        </span>
      </div>

      {/* confidence bar */}
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 4,
            fontSize: "0.78rem",
            opacity: 0.75,
          }}
        >
          <span>Confidence</span>
          <span style={{ color, fontWeight: 600 }}>{pctVal}</span>
        </div>
        <div
          style={{
            height: 7,
            borderRadius: 99,
            background: "rgba(255,255,255,0.1)",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${widthPct}%`,
              height: "100%",
              background: color,
              borderRadius: 99,
              transition: "width 0.6s ease",
            }}
          />
        </div>
      </div>

      {/* low confidence warning */}
      {is_low_confidence && (
        <div
          className="error-banner"
          style={{
            background: "rgba(234,179,8,0.12)",
            border: "1px solid rgba(234,179,8,0.4)",
            borderRadius: 6,
            padding: "0.4rem 0.7rem",
            color: "#fde047",
            fontSize: "0.8rem",
            display: "flex",
            alignItems: "center",
            gap: "0.4rem",
          }}
        >
          ⚠️ AI kurang yakin dengan prediksi ini
        </div>
      )}

      {/* top-3 alternatives */}
      {top3.length > 0 && (
        <div>
          <div style={{ fontSize: "0.75rem", opacity: 0.55, marginBottom: "0.35rem", textTransform: "uppercase", letterSpacing: 0.8 }}>
            Top 3 Alternatif
          </div>
          <div className="distribution-list" style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
            {top3.map((item, idx) => {
              const itemColor = confidenceColor(item.confidence);
              const itemWidth = Math.min(100, Math.round(item.confidence * 100));
              return (
                <div
                  key={idx}
                  className="distribution-row"
                  style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
                >
                  <span
                    className="distribution-name"
                    style={{ width: 110, fontSize: "0.78rem", opacity: 0.85, flexShrink: 0, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
                    title={labelOf(item.label)}
                  >
                    {labelOf(item.label)}
                  </span>
                  <div
                    className="distribution-track"
                    style={{ flex: 1, height: 5, borderRadius: 99, background: "rgba(255,255,255,0.08)", overflow: "hidden" }}
                  >
                    <div
                      className="distribution-fill"
                      style={{ width: `${itemWidth}%`, height: "100%", background: itemColor, borderRadius: 99 }}
                    />
                  </div>
                  <span
                    className="distribution-val"
                    style={{ width: 44, textAlign: "right", fontSize: "0.75rem", color: itemColor, flexShrink: 0 }}
                  >
                    {pct(item.confidence)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* feedback section */}
      {!fs.submitted ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          <div style={{ fontSize: "0.75rem", opacity: 0.55 }}>Apakah prediksi ini benar?</div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <button
              className="btn-secondary"
              onClick={() => onFeedback(messageId, true)}
              style={{
                fontSize: "0.8rem",
                padding: "0.3rem 0.8rem",
                border: "1px solid #22c55e",
                color: "#22c55e",
                background: "rgba(34,197,94,0.08)",
                borderRadius: 6,
                cursor: "pointer",
              }}
            >
              ✓ Benar
            </button>
            <button
              className="btn-secondary"
              onClick={() =>
                onFeedback(messageId, false, null, true /* showInput */)
              }
              style={{
                fontSize: "0.8rem",
                padding: "0.3rem 0.8rem",
                border: "1px solid #ef4444",
                color: "#ef4444",
                background: "rgba(239,68,68,0.08)",
                borderRadius: 6,
                cursor: "pointer",
              }}
            >
              ✗ Salah
            </button>
          </div>

          {/* label correction input */}
          {fs.showing && (
            <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
              <input
                type="text"
                placeholder="Label yang benar..."
                value={fs.labelInput}
                onChange={(e) =>
                  onFeedback(messageId, false, e.target.value, true /* showInput */, true /* inputOnly */)
                }
                style={{
                  flex: 1,
                  minWidth: 120,
                  padding: "0.35rem 0.6rem",
                  background: "var(--card-bg)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  color: "var(--text-primary)",
                  fontFamily: "inherit",
                  fontSize: "0.82rem",
                }}
              />
              <button
                className="btn-shimmer-cta"
                disabled={!fs.labelInput.trim()}
                onClick={() => onFeedback(messageId, false, fs.labelInput)}
                style={{ fontSize: "0.8rem", padding: "0.35rem 0.9rem" }}
              >
                Kirim Koreksi
              </button>
            </div>
          )}
        </div>
      ) : (
        <div
          style={{
            fontSize: "0.78rem",
            color: "#22c55e",
            opacity: 0.9,
            display: "flex",
            alignItems: "center",
            gap: "0.3rem",
          }}
        >
          ✓ Terima kasih atas feedback Anda!
        </div>
      )}
    </div>
  );
}

function AssistantBubble({ message, onFeedback, feedbackState }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: "0.5rem", maxWidth: "85%" }}>
      {/* avatar */}
      <div
        style={{
          width: 28,
          height: 28,
          borderRadius: "50%",
          background: "var(--accent)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 14,
          flexShrink: 0,
          marginTop: 2,
        }}
      >
        🤖
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.1rem", flex: 1 }}>
        <div
          style={{
            background: "var(--card-bg)",
            border: "1px solid var(--border)",
            borderRadius: "0 12px 12px 12px",
            padding: "0.6rem 0.9rem",
            wordBreak: "break-word",
            lineHeight: 1.6,
            fontSize: "0.9rem",
            color: "var(--text-primary)",
          }}
        >
          {message.text}

          {message.prediction && (
            <PredictionCard
              prediction={message.prediction}
              messageId={message.id}
              feedbackState={feedbackState}
              onFeedback={onFeedback}
            />
          )}
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────── main component ──────────────────── */

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [feedbackStates, setFeedbackStates] = useState({});

  const fileInputRef = useRef(null);
  const bottomRef = useRef(null);
  const messagesEndRef = useRef(null);

  /* ── welcome message on mount ── */
  useEffect(() => {
    const welcome = {
      id: uid(),
      role: "assistant",
      text: "Halo! Saya Arun-D AI Assistant 👋\n\nSaya dapat membantu Anda mengklasifikasikan gambar penyakit daun tanaman. Unggah gambar atau tanyakan sesuatu untuk memulai.",
      prediction: null,
    };
    setMessages([welcome]);
  }, []);

  /* ── auto-scroll to bottom ── */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  /* ── file selection ── */
  function handleFileSelect(file) {
    if (!file) return;
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }

  /* ── send message ── */
  async function sendMessage() {
    if (isLoading) return;
    if (!inputText.trim() && !selectedFile) return;

    const userMsg = {
      id: uid(),
      role: "user",
      text: inputText.trim(),
      imageUrl: previewUrl,
    };

    // append user bubble
    setMessages((prev) => [...prev, userMsg]);
    const sentText = inputText;
    const sentFile = selectedFile;
    setInputText("");
    setSelectedFile(null);
    setPreviewUrl(null);
    setIsLoading(true);

    // insert typing indicator placeholder id (we replace it on response)
    const assistantId = uid();

    try {
      const data = await api.chat(sentText, sentFile);

      let responseText = "";
      let prediction = null;

      if (data.response) {
        responseText = data.response;
      }
      if (data.prediction) {
        prediction = data.prediction;
        if (!responseText) {
          responseText = `Saya telah menganalisis gambar Anda. Berikut hasilnya:`;
        }
      }
      if (!responseText) {
        responseText = "Maaf, saya tidak mendapatkan respons dari server.";
      }

      const assistantMsg = {
        id: assistantId,
        role: "assistant",
        text: responseText,
        prediction,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errMsg = {
        id: assistantId,
        role: "assistant",
        text: `⚠️ Terjadi kesalahan: ${err?.message ?? "Gagal menghubungi server."}`,
        prediction: null,
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
    }
  }

  /* ── submit feedback ── */
  async function submitFeedback(
    messageId,
    isCorrect,
    correctLabel = "",
    showInput = false,
    inputOnly = false
  ) {
    // if just toggling the "Salah" panel or typing in the input, update UI state only
    if (showInput && !inputOnly) {
      // "Salah" button clicked — reveal input
      setFeedbackStates((prev) => ({
        ...prev,
        [messageId]: {
          showing: true,
          labelInput: prev[messageId]?.labelInput ?? "",
          submitted: false,
        },
      }));
      return;
    }

    if (inputOnly) {
      // user is typing the correction label
      setFeedbackStates((prev) => ({
        ...prev,
        [messageId]: {
          ...prev[messageId],
          labelInput: correctLabel ?? "",
        },
      }));
      return;
    }

    // actual submission
    const msg = messages.find((m) => m.id === messageId);
    if (!msg) return;
    const pred = msg.prediction;
    if (!pred) return;

    const original_label = pred.predicted;
    const finalCorrect = isCorrect ? original_label : (correctLabel || "").trim();

    try {
      await api.submitFeedback({
        prediction_id: pred.prediction_id ?? null,
        original_label,
        correct_label: finalCorrect,
        is_correct: isCorrect,
        image_path: null,
      });
    } catch (_) {
      // best-effort; still mark submitted so user sees confirmation
    }

    setFeedbackStates((prev) => ({
      ...prev,
      [messageId]: {
        showing: false,
        labelInput: "",
        submitted: true,
      },
    }));
  }

  /* ── render ── */
  return (
    <>
      {/* keyframe injection */}
      <style>{`
        @keyframes typingBounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.6; }
          40%            { transform: translateY(-6px); opacity: 1; }
        }
      `}</style>

      <div
        className="bento-card"
        style={{
          display: "flex",
          flexDirection: "column",
          height: "70vh",
        }}
      >
        {/* header */}
        <div className="card-header">
          <span className="card-label">Chat AI</span>
          <span className="card-label">Image Classifier</span>
        </div>

        {/* messages container */}
        <div
          ref={messagesEndRef}
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "1rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.75rem",
          }}
        >
          {messages.map((msg) =>
            msg.role === "user" ? (
              <UserBubble key={msg.id} message={msg} />
            ) : (
              <AssistantBubble
                key={msg.id}
                message={msg}
                onFeedback={submitFeedback}
                feedbackState={feedbackStates[msg.id]}
              />
            )
          )}
          {isLoading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        {/* input bar */}
        <div
          style={{
            borderTop: "1px solid var(--border)",
            padding: "0.75rem",
            display: "flex",
            gap: "0.5rem",
            alignItems: "flex-end",
          }}
        >
          {previewUrl && (
            <div style={{ position: "relative", flexShrink: 0 }}>
              <img
                src={previewUrl}
                alt="preview"
                style={{
                  width: 40,
                  height: 40,
                  objectFit: "cover",
                  borderRadius: 4,
                  border: "1px solid var(--border)",
                }}
              />
              <button
                onClick={() => {
                  setSelectedFile(null);
                  setPreviewUrl(null);
                }}
                style={{
                  position: "absolute",
                  top: -6,
                  right: -6,
                  width: 16,
                  height: 16,
                  borderRadius: "50%",
                  background: "#ef4444",
                  color: "#fff",
                  border: "none",
                  cursor: "pointer",
                  fontSize: 10,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  lineHeight: 1,
                  padding: 0,
                }}
                title="Hapus gambar"
              >
                ✕
              </button>
            </div>
          )}

          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Ketik pesan atau tanya tentang gambar..."
            style={{
              flex: 1,
              resize: "none",
              minHeight: 40,
              maxHeight: 120,
              padding: "0.5rem",
              background: "var(--card-bg)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              color: "var(--text-primary)",
              fontFamily: "inherit",
              fontSize: "0.9rem",
              lineHeight: 1.5,
              outline: "none",
            }}
            rows={1}
          />

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFileSelect(e.target.files[0])}
          />

          <button
            className="btn-secondary"
            onClick={() => fileInputRef.current?.click()}
            title="Unggah gambar"
            style={{ flexShrink: 0 }}
          >
            📷
          </button>

          <button
            className="btn-shimmer-cta"
            onClick={sendMessage}
            disabled={isLoading && !inputText && !selectedFile}
            style={{ flexShrink: 0 }}
          >
            Kirim
          </button>
        </div>
      </div>
    </>
  );
}
