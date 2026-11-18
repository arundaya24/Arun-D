const BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error("Server tidak dapat dihubungi. Pastikan backend berjalan di port 8000.");
  }
  if (!res.ok) {
    let detail = `Permintaan gagal (${res.status}).`;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      detail = `Permintaan gagal (${res.status}).`;
    }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  predict(file) {
    const form = new FormData();
    form.append("file", file);
    return request("/api/predict", { method: "POST", body: form });
  },
  history: (limit = 20, offset = 0) => request(`/api/history?limit=${limit}&offset=${offset}`),
  removeHistory: (id) => request(`/api/history/${id}`, { method: "DELETE" }),
  stats: () => request("/api/stats"),
  modelInfo: () => request("/api/model/info"),
  modelMetrics: () => request("/api/model/metrics"),
};

export const LABEL_ID = {
  airplane: "Pesawat",
  automobile: "Mobil",
  bird: "Burung",
  cat: "Kucing",
  deer: "Rusa",
  dog: "Anjing",
  frog: "Katak",
  horse: "Kuda",
  ship: "Kapal",
  truck: "Truk",
};

export const labelOf = (l) => LABEL_ID[l] || l;
export const pct = (v) => `${(v * 100).toFixed(1)}%`;
