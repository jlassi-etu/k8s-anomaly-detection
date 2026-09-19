#!/usr/bin/env python3
"""
Pousse les événements d'anomalies (anomaly_events.csv, généré par
chaos_orchestrator.py) comme annotations Grafana, pour les voir apparaître
comme des marqueurs verticaux colorés sur les graphiques du dashboard.

Prérequis: un token d'API Grafana (Grafana > Administration > Service accounts
> Add service account > rôle Editor > Add service account token).

Usage:
    python3 push_annotations_to_grafana.py --grafana-url http://localhost:3000 --token glsa_xxx
"""

import argparse
import csv
from datetime import datetime, timezone

import requests

# Une couleur (tag) par type d'anomalie pour les distinguer visuellement dans Grafana
ANOMALY_TAGS = {
    "cpu_spike": ["anomaly", "cpu_spike"],
    "memory_leak": ["anomaly", "memory_leak"],
    "resource_saturation": ["anomaly", "resource_saturation"],
    "network_anomaly": ["anomaly", "network_anomaly"],
    "abnormal_drop": ["anomaly", "abnormal_drop"],
}


def to_epoch_ms(iso_timestamp):
    dt = datetime.fromisoformat(iso_timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def load_events(csv_path):
    """Regroupe les lignes start/end du CSV en intervalles (type, début, fin)."""
    starts = {}
    intervals = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = row["type"]
            if row["action"] == "start":
                starts[key] = row["timestamp_utc"]
            elif row["action"] == "end" and key in starts:
                intervals.append((key, starts.pop(key), row["timestamp_utc"]))
    return intervals


def push_annotation(grafana_url, token, anomaly_type, start_iso, end_iso):
    payload = {
        "time": to_epoch_ms(start_iso),
        "timeEnd": to_epoch_ms(end_iso),
        "tags": ANOMALY_TAGS.get(anomaly_type, ["anomaly"]),
        "text": f"Anomalie injectée: {anomaly_type}",
    }
    resp = requests.post(
        f"{grafana_url}/api/annotations",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if resp.status_code >= 300:
        print(f"  [ERREUR] {anomaly_type} ({start_iso}): {resp.status_code} {resp.text}")
    else:
        print(f"  OK: {anomaly_type} de {start_iso} à {end_iso}")


def run(csv_path, grafana_url, token, dry_run):
    intervals = load_events(csv_path)
    print(f"{len(intervals)} fenêtres d'anomalies trouvées dans {csv_path}.")

    if dry_run:
        for anomaly_type, start, end in intervals:
            print(f"  [dry-run] {anomaly_type}: {start} -> {end}")
        return

    for anomaly_type, start, end in intervals:
        push_annotation(grafana_url, token, anomaly_type, start, end)

    print("\nTerminé. Les annotations sont visibles sur le dashboard "
          "(activez-les via la légende 'Annotations & Alerts' si besoin).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pousse les événements d'anomalies comme annotations Grafana")
    parser.add_argument("--events-csv", default="anomaly_events.csv")
    parser.add_argument("--grafana-url", default="http://localhost:3000")
    parser.add_argument("--token", required=False, help="Token d'API Grafana (service account)")
    parser.add_argument("--dry-run", action="store_true", help="Afficher les événements sans les envoyer")
    args = parser.parse_args()

    if not args.dry_run and not args.token:
        raise SystemExit("Erreur: --token est requis sauf en mode --dry-run")

    run(args.events_csv, args.grafana_url, args.token, args.dry_run)
