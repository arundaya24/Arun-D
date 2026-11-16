import { useState } from "react";
import Predict from "./pages/Predict.jsx";
import History from "./pages/History.jsx";
import Dashboard from "./pages/Dashboard.jsx";

const TABS = [
  { id: "predict", label: "Klasifikasi", view: Predict },
  { id: "history", label: "Riwayat", view: History },
  { id: "dashboard", label: "Dashboard", view: Dashboard },
];

export default function App() {
  const [tab, setTab] = useState("predict");
  const View = TABS.find((t) => t.id === tab).view;

  return (
    <div className="shell">
      <header className="masthead">
        <div className="brand">
          <span className="brand-sun" aria-hidden="true" />
          <div>
            <h1>ARUNA</h1>
            <p>Mengenali objek dari gambar dengan jaringan saraf buatan sendiri.</p>
          </div>
        </div>
        <nav aria-label="Navigasi utama">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={t.id === tab ? "tab active" : "tab"}
              aria-current={t.id === tab ? "page" : undefined}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        <View />
      </main>
    </div>
  );
}
