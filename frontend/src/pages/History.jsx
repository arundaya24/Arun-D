import { useCallback, useEffect, useState } from "react";
import { api, labelOf, pct } from "../api.js";

const PAGE = 15;

export default function History() {
  const [data, setData] = useState({ total: 0, items: [] });
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      setError("");
      setData(await api.history(PAGE, offset));
    } catch (e) {
      setError(e.message);
    }
  }, [offset]);

  useEffect(() => { load(); }, [load]);

  async function remove(id) {
    await api.removeHistory(id);
    load();
  }

  if (error) return <p className="error">{error}</p>;
  if (!data.items.length && offset === 0)
    return <p className="muted empty">Belum ada prediksi. Unggah gambar di tab Klasifikasi untuk memulai.</p>;

  return (
    <section>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Waktu</th><th>Berkas</th><th>Prediksi</th><th>Keyakinan</th><th>Waktu proses</th><th /></tr>
          </thead>
          <tbody>
            {data.items.map((r) => (
              <tr key={r.id}>
                <td>{new Date(r.created_at).toLocaleString("id-ID")}</td>
                <td className="file">{r.filename}</td>
                <td><b>{labelOf(r.predicted)}</b></td>
                <td>{pct(r.confidence)}</td>
                <td>{r.inference_ms} ms</td>
                <td><button className="link" onClick={() => remove(r.id)}>Hapus</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <button className="ghost" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Sebelumnya</button>
        <span>{offset + 1}–{Math.min(offset + PAGE, data.total)} dari {data.total}</span>
        <button className="ghost" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>Berikutnya</button>
      </div>
    </section>
  );
}
