#!/usr/bin/env python3
"""
Entraîne un classifieur supervisé multi-classes pour identifier LEQUEL des 5
types d'anomalies est en cours (cpu_spike, memory_leak, resource_saturation,
network_anomaly, abnormal_drop), ou si tout est normal.

Complémentaire à l'Isolation Forest (02_train_isolation_forest.py) qui dit
seulement "anomalie oui/non" sans préciser le type. Ce script utilise
directement la colonne "label" (connue car les anomalies ont été injectées de
façon contrôlée) comme cible d'un RandomForestClassifier.

Usage:
    python3 05_train_type_classifier.py
"""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ID_COLS = ["timestamp_utc", "pod", "label", "is_anomaly"]


def run(input_path, model_output, test_size, class_weight_arg):
    print(f"Chargement de {input_path}...")
    df = pd.read_csv(input_path, parse_dates=["timestamp_utc"])

    feature_cols = [c for c in df.columns if c not in ID_COLS]
    X = df[feature_cols].fillna(0)
    y = df["label"]

    print(f"  {len(df)} lignes, {len(feature_cols)} features.")
    print("  Répartition des classes:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print("\nEntraînement du RandomForestClassifier (multi-classes)...")
    class_weight = None if class_weight_arg == "none" else class_weight_arg
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Évaluation sur le jeu de test ===")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion (lignes=vérité, colonnes=prédiction):")
    print("Classes:", list(model.classes_))
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 10 features les plus importantes pour distinguer les types:")
    print(importances.sort_values(ascending=False).head(10))

    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "classes": list(model.classes_)},
        model_output,
    )
    print(f"\nModèle sauvegardé dans {model_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du classifieur de type d'anomalie")
    parser.add_argument("--input", default="features_dataset.csv")
    parser.add_argument("--model-output", default="anomaly_type_classifier.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--class-weight",
        default="none",
        choices=["none", "balanced"],
        help="'none' (défaut, recommandé pour le monitoring live: moins de faux positifs) "
             "ou 'balanced' (meilleur rappel sur les classes rares, mais plus de faux positifs observés en pratique)",
    )
    args = parser.parse_args()

    run(args.input, args.model_output, args.test_size, args.class_weight)
