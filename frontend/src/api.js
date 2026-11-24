const BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options = {}) {
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

  history() {
    return request("/api/history");
  },

  removeHistory(id) {
    return request(`/api/history/${id}`, { method: "DELETE" });
  },

  stats() {
    return request("/api/stats");
  },

  modelInfo() {
    return request("/api/model/info");
  },

  modelMetrics() {
    return request("/api/model/metrics");
  },

  /**
   * POST /api/chat: multipart form with "message" and optional "file"
   * @param {string} message
   * @param {File|null} file
   */
  chat(message, file = null) {
    const form = new FormData();
    form.append("message", message);
    if (file) form.append("file", file);
    return request("/api/chat", { method: "POST", body: form });
  },

  /**
   * @param {object} data
   */
  submitFeedback(data) {
    return request("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  },

  chatStats() {
    return request("/api/feedback/stats");
  },

  retrainModel() {
    return request("/api/retrain", { method: "POST" });
  },

  retrainStatus() {
    return request("/api/retrain/status");
  },
};


export const LABEL_ID = {
  human: "Manusia",
  person: "Manusia",

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

  tench: "Ikan Tench",
  goldfish: "Ikan Mas",
  great_white_shark: "Hiu Putih Besar",
  tiger_shark: "Hiu Macan",
  hammerhead: "Hiu Martil",
  electric_ray: "Pari Listrik",
  stingray: "Pari",

  cock: "Ayam Jago",
  hen: "Ayam Betina",
  ostrich: "Burung Unta",
  brambling: "Brambling",
  goldfinch: "Kenari Emas",
  house_finch: "Pipit Rumah",
  junco: "Junco",
  indigo_bunting: "Bunting Nila",
  robin: "Robin",
  bulbul: "Burung Merbah",
  jay: "Jay",
  magpie: "Murai",
  chickadee: "Chickadee",
  water_ouzel: "Penceburng Air",
  kite: "Layang-layang",
  bald_eagle: "Elang Botak",
  vulture: "Burung Nasar",
  great_grey_owl: "Burung Hantu Abu-abu",

  european_fire_salamander: "Salamander Api",
  common_newt: "Kadal Air",
  eft: "Eft",
  spotted_salamander: "Salamander Berbintik",
  axolotl: "Axolotl",
  bullfrog: "Katak Lembu",
  tree_frog: "Katak Pohon",
  tailed_frog: "Katak Berekor",

  loggerhead: "Penyu Kepala Besar",
  leatherback_turtle: "Penyu Belimbing",
  mud_turtle: "Kura-kura Lumpur",
  terrapin: "Terrapin",
  box_turtle: "Kura-kura Kotak",

  banded_gecko: "Gecko Berpita",
  common_iguana: "Iguana Biasa",
  whiptail: "Kadal Cambuk",
  agama: "Agama",
  frilled_lizard: "Kadal Berumbai",
  alligator_lizard: "Kadal Buaya",
  Gila_monster: "Gila Monster",
  green_lizard: "Kadal Hijau",
  African_chameleon: "Bunglon Afrika",
  Komodo_dragon: "Komodo",
  African_crocodile: "Buaya Afrika",
  American_alligator: "Alligator Amerika",
  triceratops: "Triceratops",

  thunder_snake: "Ular Guntur",
  ringneck_snake: "Ular Leher Cincin",
  hognose_snake: "Ular Moncong Babi",
  green_snake: "Ular Hijau",
  king_snake: "Ular Raja",
  garter_snake: "Ular Garter",
  water_snake: "Ular Air",
  vine_snake: "Ular Pohon",
  night_snake: "Ular Malam",
  boa_constrictor: "Boa Konstriktor",
  rock_python: "Piton Batu",
  Indian_cobra: "Kobra India",
  green_mamba: "Mamba Hijau",
  sea_snake: "Ular Laut",
  horned_viper: "Viper Bertanduk",
  diamondback: "Derik Berlian",
  sidewinder: "Sidewinder",

  trilobite: "Trilobit",
  harvestman: "Harvestman",
  scorpion: "Kalajengking",
  black_and_gold_garden_spider: "Laba-laba Kebun",
  barn_spider: "Laba-laba Lumbung",
  garden_spider: "Laba-laba Taman",
  black_widow: "Black Widow",
  tarantula: "Tarantula",
  wolf_spider: "Laba-laba Serigala",
  tick: "Kutu",
  centipede: "Lipan",

  tabby: "Kucing Tabby",
  tiger_cat: "Kucing Harimau",
  persian_cat: "Kucing Persia",
  siamese_cat: "Kucing Siam",
  Egyptian_cat: "Kucing Mesir",

  cougar: "Puma",
  lynx: "Lynx",
  leopard: "Macan Tutul",
  snow_leopard: "Macan Tutul Salju",
  jaguar: "Jaguar",
  lion: "Singa",
  tiger: "Harimau",
  cheetah: "Cheetah",

  brown_bear: "Beruang Coklat",
  American_black_bear: "Beruang Hitam Amerika",
  ice_bear: "Beruang Es/Polar",
  sloth_bear: "Beruang Sloth",

  mongoose: "Musang India",
  meerkat: "Meerkat",
  wood_rabbit: "Kelinci Hutan",
  hare: "Kelinci Liar",
  Angora: "Angora",
  hamster: "Hamster",
  porcupine: "Landak",
  fox_squirrel: "Tupai Rubah",
  marmot: "Marmot",
  beaver: "Berang-berang",
  guinea_pig: "Marmut",
  sorrel: "Coklat Merah",

  tiger_beetle: "Kumbang Harimau",
  ladybug: "Kepik",
  ground_beetle: "Kumbang Tanah",
  long_horned_beetle: "Kumbang Tanduk Panjang",
  leaf_beetle: "Kumbang Daun",
  dung_beetle: "Kumbang Kotoran",
  rhinoceros_beetle: "Kumbang Badak",
  weevil: "Kumbang Moncong",
  fly: "Lalat",
  bee: "Lebah",
  ant: "Semut",
  grasshopper: "Belalang",
  cricket: "Jangkrik",
  walking_stick: "Serangga Tongkat",
  cockroach: "Kecoa",
  mantis: "Belalang Sembah",
  cicada: "Tonggeret",
  leafhopper: "Wereng",
  lacewing: "Sayap Renda",
  dragonfly: "Capung",
  damselfly: "Capung Jarum",
  admiral: "Kupu-kupu Admiral",
  ringlet: "Kupu-kupu Ringlet",
  monarch: "Kupu-kupu Monarch",
  cabbage_butterfly: "Kupu-kupu Kubis",
  sulphur_butterfly: "Kupu-kupu Sulfur",
  lycaenid: "Lycaenid",

  starfish: "Bintang Laut",
  sea_urchin: "Landak Laut",
  sea_cucumber: "Teripang",

  // Kategori Objek & Lingkungan Populer
  alp: "Pegunungan Salju / Gunung",
  volcano: "Gunung Berapi",
  promontory: "Tanjung Tebing",
  valley: "Lembah Pegunungan",
  lakeside: "Tepi Danau",
  seashore: "Pesisir Pantai",

  "passenger car": "Kereta Penumpang / Gerbong",
  passenger_car: "Kereta Penumpang / Gerbong",
  "electric locomotive": "Lokomotif / Kereta Listrik",
  electric_locomotive: "Lokomotif / Kereta Listrik",
  steam_locomotive: "Lokomotif Uap",
  streetcar: "Trem Listrik",

  "folding chair": "Kursi Lipat",
  folding_chair: "Kursi Lipat",
  "rocking chair": "Kursi Goyang",
  rocking_chair: "Kursi Goyang",
  pedestal: "Meja Tumpuan",
  "dining table": "Meja Makan",
  dining_table: "Meja Makan",

  bookshop: "Toko Buku",
  bookstore: "Toko Buku",
  library: "Perpustakaan",
  bookcase: "Rak Buku",

  "brain coral": "Koral Otak",
  brain_coral: "Koral Otak",

  spatula: "Spatula / Sekop Dapur",
  loudspeaker: "Speaker / Pengeras Suara",
  "cellular telephone": "Ponsel / Handphone",
  cellular_telephone: "Ponsel / Handphone",
  laptop: "Laptop Komputer",
  monitor: "Layar Monitor",

  quill: "Bulu Unggas / Pena Bulu",
  "jigsaw puzzle": "Puzzle Gambar",
  jigsaw_puzzle: "Puzzle Gambar",
  "prayer rug": "Sajadah / Karpet",
  prayer_rug: "Sajadah / Karpet",
  velvet: "Kain Beludru",

  "Great Dane": "Anjing Great Dane",
  great_dane: "Anjing Great Dane",
  tabby: "Kucing Belang (Tabby)",
  "Persian cat": "Kucing Persia / Anggora",
  persian_cat: "Kucing Persia / Anggora",
  "Siamese cat": "Kucing Siam",
  siamese_cat: "Kucing Siam",
  "Egyptian cat": "Kucing Mesir",
  egyptian_cat: "Kucing Mesir",

  "military uniform": "Seragam Militer",
  military_uniform: "Seragam Militer",
  "lab coat": "Jas Laboratorium",
  lab_coat: "Jas Laboratorium",
  television: "Televisi",
};

export const labelOf = (l) => {
  if (!l) return "";
  const key = String(l).trim();
  const lower = key.toLowerCase();
  const underscore = lower.replace(/[\s-]+/g, "_");
  const space = lower.replace(/_+/g, " ");
  return (
    LABEL_ID[key] ||
    LABEL_ID[lower] ||
    LABEL_ID[space] ||
    LABEL_ID[underscore] ||
    key
  );
};

export const pct = (v) => `${(v * 100).toFixed(1)}%`;
