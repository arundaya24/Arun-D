import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, labelOf, pct } from "../api.js";

const INK = "#1d2547", DAWN = "#d9486b", SAFFRON = "#e8a33d", SEA = "#2c7a7b";

function Figure({ label, value }) {
  return (
    <div className="figure">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [info, setInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");
  const [noModel, setNoModel] = useState(false);

  useEffect(() => {
    api.stats().then(setStats).catch((e) => setError(e.message));
    Promise.all([api.modelInfo(), api.modelMetrics()])
      .then(([i, m]) => { setInfo(i); setMetrics(m); })
      .catch((e) => (e.status === 404 ? setNoModel(true) : setError(e.message)));
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!stats) return <p className="muted">Memuat…</p>;

  const curves = metrics?.history
    ? metrics.history.train_loss.map((_, i) => ({
        epoch: i + 1,
        train_loss: metrics.history.train_loss[i],
        val_loss: metrics.history.val_loss[i],
        train_acc: metrics.history.train_acc[i],
        val_acc: metrics.history.val_acc[i],
      }))
    : [];

  return (
    <section className="dashboard">
      <h2>Penggunaan</h2>
      <div className="figures">
        <Figure label="Total prediksi" value={stats.total_predictions} />
        <Figure label="Rata-rata keyakinan" value={pct(stats.avg_confidence)} />
        <Figure label="Rata-rata waktu proses" value={`${stats.avg_inference_ms} ms`} />
      </div>

      <div className="charts two">
        <div>
          <h3>Prediksi per kelas</h3>
          {stats.by_class.length === 0 ? <p className="muted">Belum ada data.</p> : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={stats.by_class.map((c) => ({ ...c, name: labelOf(c.label) }))}>
                <CartesianGrid stroke="#d8dce8" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" name="Jumlah" fill={DAWN} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
        <div>
          <h3>14 hari terakhir</h3>
          {stats.daily.length === 0 ? <p className="muted">Belum ada data.</p> : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={stats.daily}>
                <CartesianGrid stroke="#d8dce8" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" name="Prediksi" fill={INK} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <h2>Model</h2>
      {noModel && (
        <p className="muted">
          Model belum dilatih. Jalankan <code>python -m ml.scripts.run_train</code>, lalu muat ulang halaman ini.
        </p>
      )}
      {info && (
        <>
          <div className="figures">
            <Figure label="Akurasi test" value={pct(info.test_acc)} />
            <Figure label="Loss test" value={info.test_loss} />
            <Figure label="Parameter" value={info.parameters.toLocaleString("id-ID")} />
            <Figure label="Epoch terbaik" value={`${info.best_epoch} dari ${info.epochs_run}`} />
          </div>

          <div className="charts two">
            <div>
              <h3>Loss per epoch</h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={curves}>
                  <CartesianGrid stroke="#d8dce8" vertical={false} />
                  <XAxis dataKey="epoch" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line dataKey="train_loss" name="Train" stroke={INK} dot={false} strokeWidth={2} />
                  <Line dataKey="val_loss" name="Validasi" stroke={DAWN} dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h3>Akurasi per epoch</h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={curves}>
                  <CartesianGrid stroke="#d8dce8" vertical={false} />
                  <XAxis dataKey="epoch" />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <Tooltip formatter={(v) => pct(v)} />
                  <Legend />
                  <Line dataKey="train_acc" name="Train" stroke={SEA} dot={false} strokeWidth={2} />
                  <Line dataKey="val_acc" name="Validasi" stroke={SAFFRON} dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {metrics?.per_class && (
            <>
              <h3>Kinerja per kelas (test set)</h3>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Kelas</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
                  <tbody>
                    {Object.entries(metrics.per_class).map(([k, v]) => (
                      <tr key={k}>
                        <td><b>{labelOf(k)}</b></td>
                        <td>{pct(v.precision)}</td><td>{pct(v.recall)}</td><td>{pct(v["f1-score"])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}
