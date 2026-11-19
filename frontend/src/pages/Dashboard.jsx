import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, labelOf, pct } from "../api.js";

const GRAPHITE = "#64748b";
const AMBER = "#4f46e5";
const SUBTLE_GRID = "rgba(100, 116, 139, 0.12)";

function StatItem({ title, value }) {
  return (
    <div className="stat-card">
      <span className="stat-header">{title}</span>
      <span className="stat-figure">{value}</span>
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [info, setInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [isModelMissing, setIsModelMissing] = useState(false);

  useEffect(() => {
    api.stats().then(setStats).catch((err) => setErrorMessage(err.message));
    Promise.all([api.modelInfo(), api.modelMetrics()])
      .then(([infoData, metricsData]) => {
        setInfo(infoData);
        setMetrics(metricsData);
      })
      .catch((err) => {
        if (err.status === 404) {
          setIsModelMissing(true);
        } else {
          setErrorMessage(err.message);
        }
      });
  }, []);

  if (errorMessage) return <div className="error-banner">{errorMessage}</div>;
  if (!stats) return <p className="cell-mono" style={{ color: "var(--text-tertiary)" }}>Memuat data telemetri...</p>;

  const trainingCurves = metrics?.history
    ? metrics.history.train_loss.map((_, idx) => ({
        epoch: idx + 1,
        train_loss: metrics.history.train_loss[idx],
        val_loss: metrics.history.val_loss[idx],
        train_acc: metrics.history.train_acc[idx],
        val_acc: metrics.history.val_acc[idx],
      }))
    : [];

  return (
    <section className="dashboard-container">
      <div className="section-title-wrap">
        <h2 className="section-title">Statistik Penggunaan</h2>
      </div>

      <div className="dashboard-stat-grid">
        <StatItem title="Total Prediksi" value={stats.total_predictions} />
        <StatItem title="Rata-rata Keyakinan" value={pct(stats.avg_confidence)} />
        <StatItem title="Latensi Rata-rata" value={`${stats.avg_inference_ms} ms`} />
      </div>

      <div className="charts-asymmetric">
        <div className="chart-card">
          <span className="chart-title">Distribusi Kategori</span>
          {stats.by_class.length === 0 ? (
            <p style={{ fontSize: 13, color: "var(--text-tertiary)" }}>Belum ada data rekaman.</p>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={stats.by_class.map((c) => ({ ...c, name: labelOf(c.label) }))}>
                <CartesianGrid stroke={SUBTLE_GRID} vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: GRAPHITE }} stroke="transparent" />
                <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: GRAPHITE }} stroke="transparent" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#ffffff",
                    border: "1px solid #e2e8f0",
                    borderRadius: "12px",
                    fontFamily: "var(--font-sans)",
                    fontSize: "12px",
                    boxShadow: "0 4px 20px rgba(15,23,42,0.08)",
                    color: "#1e293b",
                  }}
                />
                <Bar dataKey="count" name="Jumlah" fill={AMBER} radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="chart-card">
          <span className="chart-title">Aktivitas Harian (14 Hari)</span>
          {stats.daily.length === 0 ? (
            <p style={{ fontSize: 13, color: "var(--text-tertiary)" }}>Belum ada data rekaman.</p>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={stats.daily}>
                <CartesianGrid stroke={SUBTLE_GRID} vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: GRAPHITE }} stroke="transparent" />
                <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: GRAPHITE }} stroke="transparent" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#ffffff",
                    border: "1px solid #e2e8f0",
                    borderRadius: "12px",
                    fontFamily: "var(--font-sans)",
                    fontSize: "12px",
                    boxShadow: "0 4px 20px rgba(15,23,42,0.08)",
                    color: "#1e293b",
                  }}
                />
                <Bar dataKey="count" name="Inferensi" fill={GRAPHITE} radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="section-title-wrap">
        <h2 className="section-title">Kinerja Model</h2>
      </div>

      {isModelMissing && (
        <div className="stat-card">
          <p style={{ fontSize: 13, color: "var(--text-secondary)" }}>
            Model belum dimuat atau belum dilatih. Jalankan perintah latihan model di backend untuk menampilkan metrik.
          </p>
        </div>
      )}

      {info && (
        <>
          <div className="dashboard-stat-grid">
            <StatItem title="Akurasi Uji" value={pct(info.test_acc)} />
            <StatItem title="Loss Uji" value={info.test_loss} />
            <StatItem title="Total Parameter" value={info.parameters.toLocaleString("id-ID")} />
            <StatItem title="Epoch Terbaik" value={`${info.best_epoch} / ${info.epochs_run}`} />
          </div>

          <div className="charts-asymmetric">
            <div className="chart-card">
              <span className="chart-title">Kurva Loss (Training vs Validasi)</span>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={trainingCurves}>
                  <CartesianGrid stroke={SUBTLE_GRID} vertical={false} />
                  <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: GRAPHITE }} stroke="transparent" />
                  <YAxis tick={{ fontSize: 11, fill: GRAPHITE }} stroke="transparent" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#ffffff",
                      border: "1px solid #e2e8f0",
                      borderRadius: "12px",
                      fontFamily: "var(--font-sans)",
                      fontSize: "12px",
                      boxShadow: "0 4px 20px rgba(15,23,42,0.08)",
                      color: "#1e293b",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px", fontFamily: "var(--font-sans)" }} />
                  <Line dataKey="train_loss" name="Training" stroke={GRAPHITE} dot={false} strokeWidth={2} />
                  <Line dataKey="val_loss" name="Validasi" stroke={AMBER} dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <span className="chart-title">Kurva Akurasi</span>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={trainingCurves}>
                  <CartesianGrid stroke={SUBTLE_GRID} vertical={false} />
                  <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: GRAPHITE }} stroke="transparent" />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fontSize: 11, fill: GRAPHITE }}
                    stroke="transparent"
                    tickFormatter={(val) => `${Math.round(val * 100)}%`}
                  />
                  <Tooltip
                    formatter={(val) => pct(val)}
                    contentStyle={{
                      backgroundColor: "#ffffff",
                      border: "1px solid #e2e8f0",
                      borderRadius: "12px",
                      fontFamily: "var(--font-sans)",
                      fontSize: "12px",
                      boxShadow: "0 4px 20px rgba(15,23,42,0.08)",
                      color: "#1e293b",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px", fontFamily: "var(--font-sans)" }} />
                  <Line dataKey="train_acc" name="Training" stroke={GRAPHITE} dot={false} strokeWidth={2} />
                  <Line dataKey="val_acc" name="Validasi" stroke={AMBER} dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {metrics?.per_class && (
            <div className="chart-card">
              <span className="chart-title">Evaluasi Per Kategori</span>
              <div className="data-table-shell">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Kategori</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1-Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics.per_class).map(([className, scoreData]) => (
                      <tr key={className}>
                        <td className="cell-primary">{labelOf(className)}</td>
                        <td className="cell-mono">{pct(scoreData.precision)}</td>
                        <td className="cell-mono">{pct(scoreData.recall)}</td>
                        <td className="cell-mono">{pct(scoreData["f1-score"])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
