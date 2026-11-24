import { useEffect, useState } from "react";
import Predict from "./pages/Predict.jsx";
import Chat from "./pages/Chat.jsx";
import History from "./pages/History.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import LiquidCanvas from "./components/LiquidCanvas.jsx";

const NAVIGATION_TABS = [
  { id: "predict", label: "Deteksi Gambar", view: Predict },
  { id: "chat", label: "Tanya AI", view: Chat },
  { id: "history", label: "Riwayat", view: History },
  { id: "dashboard", label: "Statistik", view: Dashboard },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const [theme, setTheme] = useState(() => {
    return document.documentElement.dataset.theme === "light" ? "light" : "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    document.documentElement.style.colorScheme = theme;
    try {
      localStorage.setItem("arun_d_theme", theme);
    } catch {}
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  const CurrentView = NAVIGATION_TABS.find((item) => item.id === activeTab).view;

  return (
    <>
      <LiquidCanvas theme={theme} />
      <div className="app-container">
        <header className="masthead">
          <div className="brand-section">
            <div className="brand-identity">
              <span className="brand-badge">AI Vision</span>
              <h1 className="brand-title">Arun-D</h1>
            </div>
            <p className="brand-description">
              Asisten cerdas untuk mengenali objek dan orang pada foto, lengkap dengan fitur tanya-jawab dan koreksi label langsung.
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
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

            <button
              type="button"
              className="theme-toggle-btn"
              onClick={toggleTheme}
              title={theme === "dark" ? "Ganti ke Mode Terang" : "Ganti ke Mode Gelap"}
              aria-label="Toggle Theme"
            >
              {theme === "dark" ? (
                <>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="5" />
                    <line x1="12" y1="1" x2="12" y2="3" />
                    <line x1="12" y1="21" x2="12" y2="23" />
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                    <line x1="1" y1="12" x2="3" y2="12" />
                    <line x1="21" y1="12" x2="23" y2="12" />
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                  </svg>
                  <span className="theme-toggle-label">Terang</span>
                </>
              ) : (
                <>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                  </svg>
                  <span className="theme-toggle-label">Gelap</span>
                </>
              )}
            </button>
          </div>
        </header>

        <main className="content-shell">
          <CurrentView />
        </main>
      </div>
    </>
  );
}
