#!/usr/bin/env python3
"""
Orchestrateur d'injection d'anomalies pour le cluster Kind.

Applique les manifestes Chaos Mesh selon le planning défini dans
anomaly_schedule.json, à des horaires précis relatifs au démarrage du script.
Maintient un fichier d'état (anomaly_state.json) lu en temps réel par
collect_metrics.py pour labelliser automatiquement le dataset, et journalise
chaque événement (début/fin réels) dans anomaly_events.csv.

Usage:
    python3 chaos_orchestrator.py
    python3 chaos_orchestrator.py --schedule anomaly_schedule.json --state-file anomaly_state.json
"""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_schedule(path):
    with open(path, "r") as f:
        events = json.load(f)
    # Tri par heure de démarrage pour garantir l'ordre d'exécution
    return sorted(events, key=lambda e: e["start_min"])


def write_state(state_file, active, anomaly_type, event_start=None):
    state = {
        "active": active,
        "type": anomaly_type if active else "normal",
        "event_start": event_start,
        "updated_at": utc_now_iso(),
    }
    with open(state_file, "w") as f:
        json.dump(state, f)


def init_events_log(log_file):
    log_path = Path(log_file)
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "action", "timestamp_utc", "manifest"])


def log_event(log_file, anomaly_type, action, manifest):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([anomaly_type, action, utc_now_iso(), manifest])


def apply_manifest(manifest_path):
    print(f"  -> kubectl apply -f {manifest_path}")
    subprocess.run(["kubectl", "apply", "-f", manifest_path], check=True)


def delete_manifest(manifest_path):
    print(f"  -> kubectl delete -f {manifest_path} --ignore-not-found")
    subprocess.run(
        ["kubectl", "delete", "-f", manifest_path, "--ignore-not-found"], check=True
    )


def run(schedule_path, state_file, events_log, base_dir):
    events = load_schedule(schedule_path)
    init_events_log(events_log)
    write_state(state_file, active=False, anomaly_type="normal")

    orchestrator_start = time.monotonic()
    print(f"[{utc_now_iso()}] Orchestrateur démarré. {len(events)} anomalies planifiées.")

    for event in events:
        anomaly_type = event["type"]
        manifest = str(Path(base_dir) / event["manifest"])
        start_min = event["start_min"]
        duration_min = event["duration_min"]

        # Attendre l'heure de déclenchement (relative au démarrage de l'orchestrateur)
        target_elapsed = start_min * 60
        wait_seconds = target_elapsed - (time.monotonic() - orchestrator_start)
        if wait_seconds > 0:
            print(
                f"[{utc_now_iso()}] En attente de l'injection '{anomaly_type}' "
                f"dans {wait_seconds/60:.1f} min..."
            )
            time.sleep(wait_seconds)

        # Démarrer l'anomalie
        event_start = utc_now_iso()
        print(f"[{event_start}] DÉBUT anomalie '{anomaly_type}' ({duration_min} min)")
        apply_manifest(manifest)
        write_state(state_file, active=True, anomaly_type=anomaly_type, event_start=event_start)
        log_event(events_log, anomaly_type, "start", manifest)

        # Laisser l'anomalie active pendant sa durée
        time.sleep(duration_min * 60)

        # Arrêter l'anomalie
        event_end = utc_now_iso()
        print(f"[{event_end}] FIN anomalie '{anomaly_type}'")
        delete_manifest(manifest)
        write_state(state_file, active=False, anomaly_type="normal")
        log_event(events_log, anomaly_type, "end", manifest)

    print(f"[{utc_now_iso()}] Toutes les anomalies planifiées ont été exécutées.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrateur d'injection d'anomalies Chaos Mesh")
    parser.add_argument("--schedule", default="anomaly_schedule.json")
    parser.add_argument("--state-file", default="anomaly_state.json")
    parser.add_argument("--events-log", default="anomaly_events.csv")
    parser.add_argument("--base-dir", default=".", help="Répertoire racine où se trouvent les manifestes")
    args = parser.parse_args()

    run(args.schedule, args.state_file, args.events_log, args.base_dir)
