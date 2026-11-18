import { useState } from "react";
import Predict from "./pages/Predict.jsx";
import History from "./pages/History.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import LiquidCanvas from "./components/LiquidCanvas.jsx";

const NAVIGATION_TABS = [
  { id: "predict", label: "Inferensi", view: Predict },
  { id: "history", label: "Riwayat", view: History },
  { id: "dashboard", label: "Metrik", view: Dashboard },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const CurrentView = NAVIGATION_TABS.find((item) => item.id === activeTab).view;

  return (
    <>
      <LiquidCanvas />
      <div className="app-container">
        <header className="masthead">
          <div className="brand-section">
            <div className="brand-identity">
              <span className="brand-badge">ARUNA.SYS</span>
              <h1 className="brand-title">Aruna Vision</h1>
            </div>
            <p className="brand-description">
              Sistem klasifikasi citra berbasis jaringan saraf tiruan konvolusional dengan telemetri latensi rendah.
            </p>
          </div>
          <nav className="nav-container" aria-label="Navigasi">
            {NAVIGATION_TABS.map((tabItem) => (
              <button
                key={tabItem.id}
                className={`nav-tab ${tabItem.id === activeTab ? "active" : ""}`}
                aria-current={tabItem.id === activeTab ? "page" : undefined}
                onClick={() => setActiveTab(tabItem.id)}
              >
                {tabItem.label}
              </button>
            ))}
          </nav>
        </header>

        <main className="content-shell">
          <CurrentView />
        </main>
      </div>
    </>
  );
}
