-- Opsional: tabel dibuat otomatis oleh FastAPI saat server pertama kali jalan.
CREATE DATABASE IF NOT EXISTS arun_d CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE arun_d;

CREATE TABLE IF NOT EXISTS predictions (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  filename      VARCHAR(255) NOT NULL,
  predicted     VARCHAR(50)  NOT NULL,
  confidence    FLOAT        NOT NULL,
  top_k_json    JSON         NULL,
  inference_ms  FLOAT        NOT NULL,
  model_version VARCHAR(50)  NOT NULL,
  created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_created (created_at),
  INDEX idx_predicted (predicted)
);
