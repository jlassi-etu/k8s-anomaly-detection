#!/usr/bin/env python3
"""
Génère des visualisations à partir du dataset collecté et des résultats
des modèles (Isolation Forest + Random Forest Regressor).

Fonctionne de façon incrémentale : si seul metrics_dataset.csv existe, seules
les visualisations "dataset brut" sont produites ; les visualisations liées aux
modèles s'ajoutent automatiquement si features_dataset_scored.csv et/ou
load_predictions.csv sont présents à côté.

Usage:
    python3 04_visualize_dataset.py
    python3 04_visualize_dataset.py --output-dir figures
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif nécessaire (exécution en ligne de commande)
import matplotlib.pyplot as plt
import pandas as pd

ANOMALY_COLORS = {
    "cpu_spike": "#e74c3c",
    "memory_leak": "#9b59b6",
    "resource_saturation": "#e67e22",
    "network_anomaly": "#3498db",
    "abnormal_drop": "#c0392b",
}


def savefig(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def shade_anomaly_windows(ax, df):
    """Ombre les fenêtres temporelles où une anomalie était active, pour
    superposer visuellement le contexte sur n'importe quel graphique temporel."""
    df_sorted = df.sort_values("timestamp_utc")
    current_label = None
    start_time = None
    for _, row in df_sorted.iterrows():
        label = row["label"]
        if label != current_label:
            if current_label is not None and current_label != "normal":
                ax.axvspan(start_time, row["timestamp_utc"], color=ANOMALY_COLORS.get(current_label, "grey"), alpha=0.15)
            current_label = label
            start_time = row["timestamp_utc"]
    if current_label is not None and current_label != "normal":
        ax.axvspan(start_time, df_sorted["timestamp_utc"].iloc[-1], color=ANOMALY_COLORS.get(current_label, "grey"), alpha=0.15)


def plot_class_distribution(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["label"].value_counts()
    colors = [ANOMALY_COLORS.get(lbl, "#2ecc71") for lbl in counts.index]
    counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Répartition des labels dans le dataset")
    ax.set_ylabel("Nombre de lignes")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    savefig(fig, output_dir, "01_class_distribution.png")


def plot_cluster_cpu_timeline(raw_df, output_dir):
    node_cpu = raw_df[raw_df["metric_name"] == "node_cpu_usage_percent"]
    cluster_avg = node_cpu.groupby("timestamp_utc").agg(value=("value", "mean"), label=("label", "first")).reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cluster_avg["timestamp_utc"], cluster_avg["value"], color="#2c3e50", linewidth=1)
    shade_anomaly_windows(ax, cluster_avg)
    ax.set_title("CPU moyen des nœuds dans le temps (zones colorées = anomalie active)")
    ax.set_ylabel("CPU (%)")
    ax.set_xlabel("Temps (UTC)")
    savefig(fig, output_dir, "02_cluster_cpu_timeline.png")


def plot_pod_memory_timeline(raw_df, output_dir):
    pod_mem = raw_df[raw_df["metric_name"] == "pod_memory_working_set_bytes"]
    cluster_sum = pod_mem.groupby("timestamp_utc").agg(value=("value", "sum"), label=("label", "first")).reset_index()
    cluster_sum["value_mb"] = cluster_sum["value"] / (1024 * 1024)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cluster_sum["timestamp_utc"], cluster_sum["value_mb"], color="#8e44ad", linewidth=1)
    shade_anomaly_windows(ax, cluster_sum)
    ax.set_title("Mémoire totale des pods dans le temps (zones colorées = anomalie active)")
    ax.set_ylabel("Mémoire (Mo)")
    ax.set_xlabel("Temps (UTC)")
    savefig(fig, output_dir, "03_pod_memory_timeline.png")


def plot_boxplot_cpu_by_label(raw_df, output_dir):
    pod_cpu = raw_df[raw_df["metric_name"] == "pod_cpu_usage_cores"]
    labels_present = [l for l in ["normal"] + list(ANOMALY_COLORS.keys()) if l in pod_cpu["label"].unique()]

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [pod_cpu[pod_cpu["label"] == lbl]["value"].dropna() for lbl in labels_present]
    colors = [ANOMALY_COLORS.get(lbl, "#2ecc71") for lbl in labels_present]
    try:
        bp = ax.boxplot(data, tick_labels=labels_present, patch_artist=True, showfliers=False)
    except TypeError:  # matplotlib < 3.9 n'a pas encore 'tick_labels'
        bp = ax.boxplot(data, labels=labels_present, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Distribution du CPU par pod, selon le label")
    ax.set_ylabel("CPU (cores)")
    plt.xticks(rotation=30, ha="right")
    savefig(fig, output_dir, "04_cpu_boxplot_by_label.png")


def plot_isolation_forest_scores(scored_path, output_dir):
    if not os.path.exists(scored_path):
        print(f"  (ignoré: {scored_path} introuvable — lancez 02_train_isolation_forest.py d'abord)")
        return

    df = pd.read_csv(scored_path, parse_dates=["timestamp_utc"])
    agg = df.groupby("timestamp_utc").agg(
        score=("anomaly_score", "mean"), label=("label", "first")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(agg["timestamp_utc"], agg["score"], color="#16a085", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    shade_anomaly_windows(ax, agg)
    ax.set_title("Score d'anomalie Isolation Forest dans le temps (bas = plus anormal)")
    ax.set_ylabel("Score moyen (tous pods)")
    ax.set_xlabel("Temps (UTC)")
    savefig(fig, output_dir, "05_isolation_forest_scores.png")


def plot_load_predictions(predictions_path, output_dir):
    if not os.path.exists(predictions_path):
        print(f"  (ignoré: {predictions_path} introuvable — lancez 03_train_load_predictor.py d'abord)")
        return

    df = pd.read_csv(predictions_path, parse_dates=["timestamp_utc"])
    # Important: avec --split-mode random, les lignes du jeu de test ne sont PAS
    # contiguës dans le temps. On trie par timestamp pour l'affichage, et on
    # utilise des marqueurs (pas de ligne continue) pour ne pas relier des
    # points temporellement éloignés entre eux (effet "toile d'araignée").
    df = df.sort_values("timestamp_utc")
    is_anomaly_col = "is_anomaly" in df.columns

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(df["timestamp_utc"], df["actual_load"], label="Charge réelle",
               color="#2c3e50", s=14, alpha=0.7)
    ax.scatter(df["timestamp_utc"], df["predicted_load"], label="Charge prédite",
               color="#e74c3c", s=14, alpha=0.5, marker="x")
    ax.set_title("Random Forest Regressor — charge prédite vs réelle (ensemble de test, points non reliés)")
    ax.set_ylabel("Charge prédite")
    ax.set_xlabel("Temps (UTC)")
    ax.legend()
    savefig(fig, output_dir, "06_load_prediction_vs_actual.png")

    # Graphique de parité (prédit vs réel) : la référence standard pour juger
    # la qualité d'une régression, surtout quand le test n'est pas temporellement
    # contigu (split aléatoire). Les points doivent s'aligner sur la diagonale.
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    if is_anomaly_col:
        normal_mask = df["is_anomaly"] == 0
        ax2.scatter(df.loc[normal_mask, "actual_load"], df.loc[normal_mask, "predicted_load"],
                    color="#2ecc71", s=18, alpha=0.6, label="normal")
        ax2.scatter(df.loc[~normal_mask, "actual_load"], df.loc[~normal_mask, "predicted_load"],
                    color="#e74c3c", s=18, alpha=0.6, label="anomalie")
        ax2.legend()
    else:
        ax2.scatter(df["actual_load"], df["predicted_load"], color="#2c3e50", s=18, alpha=0.6)

    lims = [min(df["actual_load"].min(), df["predicted_load"].min()),
            max(df["actual_load"].max(), df["predicted_load"].max())]
    ax2.plot(lims, lims, color="grey", linestyle="--", linewidth=1, label="_nolegend_")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Charge réelle")
    ax2.set_ylabel("Charge prédite")
    ax2.set_title("Graphique de parité (points proches de la diagonale = bonne prédiction)")
    savefig(fig2, output_dir, "07_load_prediction_parity.png")


def run(raw_path, features_path, scored_path, predictions_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Sauvegarde des figures dans ./{output_dir}/")

    print("\n[Dataset brut]")
    raw_df = pd.read_csv(raw_path, parse_dates=["timestamp_utc"])
    plot_class_distribution(raw_df.drop_duplicates(subset=["timestamp_utc"]), output_dir)
    plot_cluster_cpu_timeline(raw_df, output_dir)
    plot_pod_memory_timeline(raw_df, output_dir)
    plot_boxplot_cpu_by_label(raw_df, output_dir)

    print("\n[Isolation Forest]")
    plot_isolation_forest_scores(scored_path, output_dir)

    print("\n[Random Forest Regressor]")
    plot_load_predictions(predictions_path, output_dir)

    print(f"\nTerminé. {len(os.listdir(output_dir))} figure(s) générée(s) dans ./{output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation du dataset et des résultats des modèles")
    parser.add_argument("--raw-input", default="metrics_dataset.csv")
    parser.add_argument("--features-input", default="features_dataset.csv")
    parser.add_argument("--scored-input", default="features_dataset_scored.csv")
    parser.add_argument("--predictions-input", default="load_predictions.csv")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    run(args.raw_input, args.features_input, args.scored_input, args.predictions_input, args.output_dir)
