#!/usr/bin/env python3
"""
Prépare le dataset de features à partir du CSV brut collecté par collect_metrics.py.

Le fichier source est au format "long" (une ligne = une métrique pour une entité
à un instant donné). Ce script le transforme en format "large" (une ligne = un pod
à un instant donné, avec toutes ses métriques en colonnes), enrichi du contexte
cluster (moyennes/nœuds au même instant) et de features temporelles (fenêtres
glissantes + valeurs décalées) pour capter les dynamiques (montée en charge,
fuite mémoire progressive, etc.).

Usage:
    python3 01_prepare_features.py
    python3 01_prepare_features.py --input metrics_dataset_clean.csv --rolling-window 5
"""

import argparse

import pandas as pd

# Métriques par pod (namespace=workload) que l'on garde comme features principales
POD_METRICS = [
    "pod_cpu_usage_cores",
    "pod_memory_working_set_bytes",
    "pod_restarts_5m",
    "pod_ready_status",
]

# Métriques par nœud, agrégées au niveau cluster à chaque instant
NODE_METRICS = [
    "node_cpu_usage_percent",
    "node_memory_usage_percent",
    "node_network_receive_bytes",
    "node_network_transmit_bytes",
    "node_network_errors",
]


def load_raw(path):
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # Unifie l'identifiant d'entité : "instance" pour les métriques nœud,
    # "pod" pour les métriques pod (une des deux colonnes est toujours vide)
    df["entity"] = df["pod"].where(df["pod"].astype(bool), df["instance"])
    return df


def build_pod_wide(df):
    pod_df = df[df["metric_name"].isin(POD_METRICS)].copy()
    wide = pod_df.pivot_table(
        index=["timestamp_utc", "entity", "label"],
        columns="metric_name",
        values="value",
        aggfunc="mean",
    ).reset_index()
    wide = wide.rename(columns={"entity": "pod"})
    return wide


def build_node_aggregates(df):
    node_df = df[df["metric_name"].isin(NODE_METRICS)].copy()
    agg_funcs = {
        "node_cpu_usage_percent": "mean",
        "node_memory_usage_percent": "mean",
        "node_network_receive_bytes": "sum",
        "node_network_transmit_bytes": "sum",
        "node_network_errors": "sum",
    }

    result = None
    for metric, func in agg_funcs.items():
        sub = node_df[node_df["metric_name"] == metric]
        if sub.empty:
            continue
        agg = sub.groupby("timestamp_utc")["value"].agg(func).rename(f"cluster_{metric}")
        result = agg.to_frame() if result is None else result.join(agg, how="outer")

    if result is None:
        return pd.DataFrame(columns=["timestamp_utc"])
    return result.reset_index()


def add_time_features(df, rolling_window):
    """Ajoute moyennes/écarts-types glissants et valeurs décalées, calculés
    séparément pour chaque pod (pour ne pas mélanger les historiques)."""
    df = df.sort_values(["pod", "timestamp_utc"]).reset_index(drop=True)

    feature_cols = [c for c in POD_METRICS if c in df.columns]

    for col in feature_cols:
        grouped = df.groupby("pod")[col]
        df[f"{col}_roll_mean"] = grouped.transform(
            lambda s: s.rolling(rolling_window, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = grouped.transform(
            lambda s: s.rolling(rolling_window, min_periods=1).std()
        )
        df[f"{col}_lag1"] = grouped.shift(1)

    # Les premières lignes de chaque pod n'ont pas de lag/std -> on comble
    df = df.sort_values(["pod", "timestamp_utc"])
    lag_std_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_roll_std")]
    df[lag_std_cols] = df.groupby("pod")[lag_std_cols].transform(lambda s: s.bfill().fillna(0))

    return df


def run(input_path, output_path, rolling_window):
    print(f"Chargement de {input_path}...")
    raw = load_raw(input_path)
    print(f"  {len(raw)} lignes brutes chargées.")

    pod_wide = build_pod_wide(raw)
    print(f"  {len(pod_wide)} lignes après pivot pod (une ligne = un pod à un instant).")

    node_agg = build_node_aggregates(raw)
    print(f"  {len(node_agg)} instants avec agrégats cluster.")

    merged = pod_wide.merge(node_agg, on="timestamp_utc", how="left")

    featured = add_time_features(merged, rolling_window)

    featured["is_anomaly"] = (featured["label"] != "normal").astype(int)

    # Réordonner les colonnes pour la lisibilité
    id_cols = ["timestamp_utc", "pod", "label", "is_anomaly"]
    other_cols = [c for c in featured.columns if c not in id_cols]
    featured = featured[id_cols + other_cols]

    featured.to_csv(output_path, index=False)
    print(f"\nFeatures sauvegardées dans {output_path} ({len(featured)} lignes, {len(featured.columns)} colonnes).")
    print("\nRépartition des labels:")
    print(featured["label"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparation des features pour le ML")
    parser.add_argument("--input", default="metrics_dataset.csv")
    parser.add_argument("--output", default="features_dataset.csv")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Taille de la fenêtre glissante en nombre d'échantillons (5 x 15s = 75s par défaut)",
    )
    args = parser.parse_args()

    run(args.input, args.output, args.rolling_window)
