"""Prediksi satu gambar: python -m ml.scripts.predict_cli path/gambar.jpg"""
import argparse
import json
from pathlib import Path

from ml.src.predict import Predictor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--artifacts", default="ml/artifacts")
    a = ap.parse_args()
    result = Predictor(a.artifacts).predict_bytes(Path(a.image).read_bytes())
    print(json.dumps(result, indent=2))
