"""Latih model: python -m ml.scripts.run_train [--config ml/configs/default.yaml]"""
import argparse

from ml.src.train import run_training

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/default.yaml")
    run_training(ap.parse_args().config)
