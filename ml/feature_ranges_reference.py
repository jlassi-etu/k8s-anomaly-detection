#!/usr/bin/env python3
"""
Affiche, pour chaque feature utilisée par les modèles, la plage de valeurs vue
comme "normale" pendant l'entraînement (min/moyenne/max/p95).

À comparer avec la sortie de `ai_inference_exporter.py --debug`, pour repérer
précisément quelle(s) feature(s) sortent de la plage connue en direct.

Usage:
    python3 feature_ranges_reference.py
    python3 feature_ranges_reference.py --input features_dataset.csv
"""

import argparse

import pandas as pd

ID_COLS = ["timestamp_utc", "pod", "label", "is_anomaly"]


def run(input_path):
    df = pd.read_csv(input_path)
    normal = df[df["label"] == "normal"]
    feature_cols = [c for c in df.columns if c not in ID_COLS]

    print(f"Plage de valeurs 'normale' vue à l'entraînement ({len(normal)} lignes):\n")
    print(f"{'Feature':<45} {'min':>14} {'moyenne':>14} {'p95':>14} {'max':>14}")
    print("-" * 105)
    for col in feature_cols:
        s = normal[col]
        print(f"{col:<45} {s.min():>14.4f} {s.mean():>14.4f} {s.quantile(0.95):>14.4f} {s.max():>14.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Référence des plages de valeurs normales par feature")
    parser.add_argument("--input", default="features_dataset.csv")
    args = parser.parse_args()

    run(args.input)
