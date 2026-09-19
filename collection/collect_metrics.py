#!/usr/bin/env python3
"""
Collecteur de métriques Prometheus pour le dataset d'entraînement.

Interroge Prometheus en boucle pendant une durée donnée (5h par défaut),
exporte chaque série temporelle scrappée dans un CSV, et labellise
automatiquement chaque ligne (normal / type d'anomalie) en lisant le fichier
d'état écrit en temps réel par chaos_orchestrator.py.

Prérequis: port-forward actif vers Prometheus
    kubectl port-forward -n monitoring svc/kube-prometheus-kube-prome-prometheus 9090:9090

Usage:
    python3 collect_metrics.py
    python3 collect_metrics.py --duration-hours 5 --interval-seconds 15
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Requêtes PromQL couvrant CPU, mémoire, réseau au niveau nœud ET pod,
# plus les redémarrages de pods (utile pour les abnormal drops).
QUERIES = {
    "node_cpu_usage_percent": (
        '100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)'
    ),
    "node_memory_usage_percent": (
        "100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))"
    ),
    "node_network_receive_bytes": (
        'sum by(instance) (rate(node_network_receive_bytes_total{device!~"lo|veth.*"}[1m]))'
    ),
    "node_network_transmit_bytes": (
        'sum by(instance) (rate(node_network_transmit_bytes_total{device!~"lo|veth.*"}[1m]))'
    ),
    "node_network_errors": (
        'sum by(instance) (rate(node_network_receive_errs_total[1m]) '
        '+ rate(node_network_transmit_errs_total[1m]))'
    ),
    "pod_cpu_usage_cores": (
        'sum by(pod, namespace) (rate(container_cpu_usage_seconds_total'
        '{namespace="workload", container!=""}[1m]))'
    ),
    "pod_memory_working_set_bytes": (
        'sum by(pod, namespace) (container_memory_working_set_bytes'
        '{namespace="workload", container!=""})'
    ),
    "pod_restarts_5m": (
        "sum by(pod, namespace) (increase(kube_pod_container_status_restarts_total"
        '{namespace="workload"}[5m]))'
    ),
    "pod_ready_status": (
        'sum by(pod, namespace) (kube_pod_status_ready{namespace="workload", condition="true"})'
    ),
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def query_prometheus(prom_url, promql):
    resp = requests.get(
        f"{prom_url}/api/v1/query", params={"query": promql}, timeout=10
    )
    resp.raise_for_status()
    result = resp.json()
    if result.get("status") != "success":
        return []
    return result["data"]["result"]


def read_anomaly_state(state_file):
    try:
        with open(state_file, "r") as f:
            state = json.load(f)
        return state.get("type", "normal"), state.get("active", False)
    except (FileNotFoundError, json.JSONDecodeError):
        # Pas d'orchestrateur en cours d'exécution -> tout est étiqueté "normal"
        return "normal", False


def init_csv(output_path):
    path = Path(output_path)
    is_new = not path.exists()
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(
            ["timestamp_utc", "metric_name", "instance", "pod", "namespace", "value", "label"]
        )
    return f, writer


def collect_once(prom_url, writer, anomaly_type):
    scrape_time = utc_now_iso()
    total_rows = 0
    for metric_name, promql in QUERIES.items():
        try:
            series_list = query_prometheus(prom_url, promql)
        except requests.RequestException as e:
            print(f"  [WARN] échec requête '{metric_name}': {e}")
            continue

        for series in series_list:
            labels = series.get("metric", {})
            value = series["value"][1]
            writer.writerow(
                [
                    scrape_time,
                    metric_name,
                    labels.get("instance", ""),
                    labels.get("pod", ""),
                    labels.get("namespace", ""),
                    value,
                    anomaly_type,
                ]
            )
            total_rows += 1
    return total_rows


def run(prom_url, duration_hours, interval_seconds, output_csv, state_file):
    end_time = time.monotonic() + duration_hours * 3600
    csv_file, writer = init_csv(output_csv)

    print(f"[{utc_now_iso()}] Début de la collecte pour {duration_hours}h "
          f"(intervalle {interval_seconds}s) -> {output_csv}")

    iteration = 0
    try:
        while time.monotonic() < end_time:
            iteration += 1
            anomaly_type, active = read_anomaly_state(state_file)
            rows = collect_once(prom_url, writer, anomaly_type)
            csv_file.flush()  # écrire immédiatement sur disque, pas seulement en mémoire

            status = f"ANOMALIE={anomaly_type}" if active else "normal"
            print(f"[{utc_now_iso()}] Itération {iteration}: {rows} lignes ({status})")

            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print(f"\n[{utc_now_iso()}] Collecte interrompue manuellement.")
    finally:
        csv_file.close()
        print(f"[{utc_now_iso()}] Collecte terminée. Fichier: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collecteur de métriques Prometheus")
    parser.add_argument("--prom-url", default="http://localhost:9090")
    parser.add_argument("--duration-hours", type=float, default=5.0)
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--output-csv", default="metrics_dataset.csv")
    parser.add_argument("--state-file", default="anomaly_state.json")
    args = parser.parse_args()

    run(
        args.prom_url,
        args.duration_hours,
        args.interval_seconds,
        args.output_csv,
        args.state_file,
    )
