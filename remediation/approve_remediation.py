#!/usr/bin/env python3
"""
Approuve et exécute l'action de remédiation en attente (générée par
remediation_controller.py --mode semi-auto).

Usage:
    python3 approve_remediation.py
    python3 approve_remediation.py --reject   # rejette l'action sans l'exécuter
"""

import argparse
import csv
import json
from pathlib import Path

from kubernetes import client, config

from remediation_controller import RemediationActions, execute_action, utc_now_iso


def run(pending_file, log_file, namespace, deployment_name, target_replicas,
        scale_up_increment, max_replicas, reject):
    path = Path(pending_file)
    if not path.exists():
        print(f"Aucune action en attente ({pending_file} introuvable).")
        return

    with open(path) as f:
        pending = json.load(f)

    print(f"Action en attente depuis {pending['timestamp_utc']}:")
    print(f"  Anomalie : {pending['anomaly_type']}")
    print(f"  Action   : {pending['action']}")
    print(f"  Description : {pending['description']}")

    if reject:
        print("\nAction rejetée (non exécutée).")
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([utc_now_iso(), pending["anomaly_type"], pending["action"],
                                     "semi-auto", "rejected", ""])
        path.unlink()
        return

    confirm = input("\nExécuter cette action maintenant ? [o/N] ").strip().lower()
    if confirm != "o":
        print("Annulé, l'action reste en attente.")
        return

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    actions = RemediationActions(
        client.AppsV1Api(), client.CoreV1Api(), namespace, deployment_name,
        target_replicas, scale_up_increment, max_replicas,
    )

    try:
        detail = execute_action(actions, pending["action"])
        print(f"\nACTION EXÉCUTÉE: {detail}")
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([utc_now_iso(), pending["anomaly_type"], pending["action"],
                                     "semi-auto", "executed_manually", detail])
    except Exception as e:
        print(f"\nÉCHEC: {e}")
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([utc_now_iso(), pending["anomaly_type"], pending["action"],
                                     "semi-auto", "failed", str(e)])

    path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Approuve/rejette une action de remédiation en attente")
    parser.add_argument("--pending-file", default="pending_remediation.json")
    parser.add_argument("--log-file", default="remediation_log.csv")
    parser.add_argument("--namespace", default="workload")
    parser.add_argument("--deployment-name", default="sample-workload")
    parser.add_argument("--target-replicas", type=int, default=6)
    parser.add_argument("--scale-up-increment", type=int, default=2)
    parser.add_argument("--max-replicas", type=int, default=10)
    parser.add_argument("--reject", action="store_true")
    args = parser.parse_args()

    run(args.pending_file, args.log_file, args.namespace, args.deployment_name,
        args.target_replicas, args.scale_up_increment, args.max_replicas, args.reject)
