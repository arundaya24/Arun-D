import { useCallback, useEffect, useState } from "react";
import { api, labelOf, pct } from "../api.js";

const PAGE_LIMIT = 15;

export default function History() {
  const [historyData, setHistoryData] = useState({ total: 0, items: [] });
  const [currentOffset, setCurrentOffset] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");

  const fetchRecords = useCallback(async () => {
    try {
      setErrorMessage("");
      const response = await api.history(PAGE_LIMIT, currentOffset);
      setHistoryData(response);
    } catch (err) {
      setErrorMessage(err.message);
    }
  }, [currentOffset]);

  useEffect(() => {
    fetchRecords();
  }, [fetchRecords]);

  async function handleRemove(id) {
    try {
      await api.removeHistory(id);
      fetchRecords();
    } catch (err) {
      setErrorMessage(err.message);
    }
  }

  if (errorMessage) return <div className="error-banner">{errorMessage}</div>;

  if (!historyData.items.length && currentOffset === 0) {
    return (
      <div className="bento-card">
        <div className="empty-state">
          <h3 className="empty-state-title">Log Riwayat Kosong</h3>
          <p className="empty-state-text">
            Belum ada catatan inferensi yang tersimpan. Unggah berkas gambar pada tab Inferensi untuk mulai mencatat.
          </p>
        </div>
      </div>
    );
  }

  return (
    <section className="bento-card" style={{ padding: 0 }}>
      <div className="data-table-shell" style={{ border: "none" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Waktu Stempel</th>
              <th>Nama Berkas</th>
              <th>Klasifikasi</th>
              <th>Tingkat Keyakinan</th>
              <th>Latensi</th>
              <th style={{ textAlign: "right" }}>Aksi</th>
            </tr>
          </thead>
          <tbody>
            {historyData.items.map((row) => (
              <tr key={row.id}>
                <td className="cell-mono">
                  {new Date(row.created_at).toLocaleString("id-ID", {
                    dateStyle: "short",
                    timeStyle: "medium",
                  })}
                </td>
                <td className="cell-mono" style={{ maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis" }}>
                  {row.filename}
                </td>
                <td className="cell-primary">{labelOf(row.predicted)}</td>
                <td className="cell-mono">{pct(row.confidence)}</td>
                <td className="cell-mono">{row.inference_ms} ms</td>
                <td style={{ textAlign: "right" }}>
                  <button
                    type="button"
                    className="btn-table-action"
                    onClick={() => handleRemove(row.id)}
                  >
                    Hapus
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="pagination-bar">
        <span className="pagination-meta">
          Data {currentOffset + 1} - {Math.min(currentOffset + PAGE_LIMIT, historyData.total)} dari {historyData.total}
        </span>
        <div className="pagination-controls">
          <button
            type="button"
            className="btn-paginate"
            disabled={currentOffset === 0}
            onClick={() => setCurrentOffset(Math.max(0, currentOffset - PAGE_LIMIT))}
          >
            Sebelumnya
          </button>
          <button
            type="button"
            className="btn-paginate"
            disabled={currentOffset + PAGE_LIMIT >= historyData.total}
            onClick={() => setCurrentOffset(currentOffset + PAGE_LIMIT)}
          >
            Berikutnya
          </button>
        </div>
      </div>
    </section>
  );
}
