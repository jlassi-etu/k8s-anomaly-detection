#!/usr/bin/env python3
"""
Contrôleur de remédiation automatique/semi-automatique.

Lit l'état d'anomalie CONFIRMÉ (déjà débruité par le debounce de
ai_inference_exporter.py) via la métrique Prometheus k8s_ai_anomaly_type_active,
et déclenche une action corrective Kubernetes adaptée dès qu'une NOUVELLE
anomalie est confirmée (déclenchement sur front montant, pas à chaque cycle).

Modes:
    --mode auto      : l'action est exécutée immédiatement.
    --mode semi-auto  : l'action est seulement journalisée comme "recommandée"
                        dans pending_remediation.json ; l'opérateur doit lancer
                        approve_remediation.py pour l'exécuter.

Toutes les actions (exécutées ou recommandées) sont journalisées dans
remediation_log.csv, et republiées comme métriques Prometheus pour être
visualisables dans Grafana.

Usage:
    python3 remediation_controller.py
    python3 remediation_controller.py --mode semi-auto
"""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from kubernetes import client, config
from prometheus_client import Counter, Gauge, start_http_server

ANOMALY_TYPES = ["cpu_spike", "memory_leak", "resource_saturation",
                  "network_anomaly", "abnormal_drop"]

G_ACTIONS_TOTAL = Counter(
    "k8s_remediation_actions_total", "Nombre d'actions correctives déclenchées", ["type", "action", "status"]
)
G_LAST_ACTION_TS = Gauge("k8s_remediation_last_action_timestamp", "Timestamp Unix de la dernière action")
G_PENDING_APPROVAL = Gauge("k8s_remediation_pending_approval", "1 si une action attend une validation manuelle")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_confirmed_anomaly_type(prom_url):
    resp = requests.get(
        f"{prom_url}/api/v1/query",
        params={"query": "k8s_ai_anomaly_type_active"},
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json()
    if result.get("status") != "success":
        return "normal"
    for series in result["data"]["result"]:
        if float(series["value"][1]) == 1.0:
            return series["metric"].get("type", "normal")
    return "normal"


def init_log(log_path):
    path = Path(log_path)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_utc", "anomaly_type", "action", "mode", "status", "detail"])


def log_action(log_path, anomaly_type, action, mode, status, detail=""):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([utc_now_iso(), anomaly_type, action, mode, status, detail])


class RemediationActions:
    """Encapsule les actions K8s réelles. Séparé pour rester testable sans cluster."""

    def __init__(self, apps_api, core_api, namespace, deployment_name, target_replicas,
                 scale_up_increment, max_replicas):
        self.apps_api = apps_api
        self.core_api = core_api
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.target_replicas = target_replicas
        self.scale_up_increment = scale_up_increment
        self.max_replicas = max_replicas
        self.scaled_up = False

    def current_replicas(self):
        dep = self.apps_api.read_namespaced_deployment(self.deployment_name, self.namespace)
        return dep.spec.replicas

    def scale_to(self, replicas):
        self.apps_api.patch_namespaced_deployment_scale(
            self.deployment_name, self.namespace, {"spec": {"replicas": replicas}}
        )
        return replicas

    def scale_up(self):
        current = self.current_replicas()
        new_count = min(current + self.scale_up_increment, self.max_replicas)
        if new_count == current:
            return f"déjà au maximum ({current} réplicas)"
        self.scale_to(new_count)
        self.scaled_up = True
        return f"{current} -> {new_count} réplicas"

    def scale_down_to_baseline(self):
        if not self.scaled_up:
            return "aucun scale-up actif, rien à annuler"
        current = self.current_replicas()
        self.scale_to(self.target_replicas)
        self.scaled_up = False
        return f"{current} -> {self.target_replicas} réplicas (retour à la ligne de base)"

    def restart_pods(self):
        pods = self.core_api.list_namespaced_pod(
            self.namespace, label_selector=f"app={self.deployment_name}"
        )
        names = [p.metadata.name for p in pods.items]
        for name in names:
            self.core_api.delete_namespaced_pod(name, self.namespace)
        return f"{len(names)} pod(s) redémarré(s): {', '.join(names)}"

    def reconcile_replicas(self):
        current = self.current_replicas()
        if current != self.target_replicas:
            self.scale_to(self.target_replicas)
            return f"réplicas corrigés: {current} -> {self.target_replicas}"
        return f"déjà au nombre nominal ({self.target_replicas} réplicas)"


def decide_action(anomaly_type):
    """Retourne (nom_action, description) sans l'exécuter."""
    mapping = {
        "cpu_spike": ("scale_up", "Augmenter les réplicas pour absorber le pic CPU"),
        "resource_saturation": ("scale_up", "Augmenter les réplicas pour absorber la saturation"),
        "memory_leak": ("restart_pods", "Redémarrer les pods pour purger la fuite mémoire"),
        "network_anomaly": ("restart_pods", "Redémarrer les pods pour réinitialiser la pile réseau"),
        "abnormal_drop": ("reconcile_replicas", "Forcer la reconciliation du nombre de réplicas"),
    }
    return mapping.get(anomaly_type, (None, None))


def execute_action(actions, action_name):
    method = getattr(actions, action_name)
    return method()


def run(prom_url, interval_seconds, mode, cooldown_seconds, namespace, deployment_name,
        target_replicas, scale_up_increment, max_replicas, log_file, pending_file, metrics_port):
    print(f"Chargement de la config Kubernetes...")
    try:
        config.load_incluster_config()
        print("  Config in-cluster chargée.")
    except config.ConfigException:
        config.load_kube_config()
        print("  Config kubeconfig locale chargée.")

    apps_api = client.AppsV1Api()
    core_api = client.CoreV1Api()
    actions = RemediationActions(
        apps_api, core_api, namespace, deployment_name, target_replicas,
        scale_up_increment, max_replicas,
    )

    init_log(log_file)
    start_http_server(metrics_port)
    print(f"Contrôleur de remédiation démarré (mode={mode}) sur http://0.0.0.0:{metrics_port}/metrics")
    print(f"Cible: Deployment '{deployment_name}' dans le namespace '{namespace}'\n")

    last_type = "normal"
    last_action_time = 0.0

    while True:
        try:
            current_type = get_confirmed_anomaly_type(prom_url)

            # --- Front montant: nouvelle anomalie confirmée ---
            if current_type != "normal" and current_type != last_type:
                elapsed = time.time() - last_action_time
                if elapsed < cooldown_seconds:
                    print(f"[{utc_now_iso()}] {current_type} détecté mais cooldown actif "
                          f"({cooldown_seconds - elapsed:.0f}s restantes), action ignorée.", flush=True)
                else:
                    action_name, description = decide_action(current_type)
                    if action_name is None:
                        print(f"[{utc_now_iso()}] Aucune action définie pour '{current_type}'.", flush=True)
                    elif mode == "auto":
                        try:
                            detail = execute_action(actions, action_name)
                            print(f"[{utc_now_iso()}] ACTION EXÉCUTÉE ({current_type}): "
                                  f"{description} -> {detail}", flush=True)
                            log_action(log_file, current_type, action_name, mode, "executed", detail)
                            G_ACTIONS_TOTAL.labels(type=current_type, action=action_name, status="executed").inc()
                        except Exception as e:
                            print(f"[{utc_now_iso()}] ÉCHEC de l'action ({current_type}): {e}", flush=True)
                            log_action(log_file, current_type, action_name, mode, "failed", str(e))
                            G_ACTIONS_TOTAL.labels(type=current_type, action=action_name, status="failed").inc()
                    else:  # semi-auto
                        pending = {
                            "timestamp_utc": utc_now_iso(),
                            "anomaly_type": current_type,
                            "action": action_name,
                            "description": description,
                        }
                        with open(pending_file, "w") as f:
                            json.dump(pending, f, indent=2)
                        print(f"[{utc_now_iso()}] ACTION RECOMMANDÉE ({current_type}): {description}\n"
                              f"  -> Lancez `python3 approve_remediation.py` pour l'exécuter, "
                              f"ou ignorez pour la laisser expirer.", flush=True)
                        log_action(log_file, current_type, action_name, mode, "pending_approval", description)
                        G_PENDING_APPROVAL.set(1)

                    last_action_time = time.time()
                    G_LAST_ACTION_TS.set(last_action_time)

            # --- Retour à la normale: annuler un éventuel scale-up ---
            elif current_type == "normal" and last_type != "normal":
                if actions.scaled_up:
                    detail = actions.scale_down_to_baseline()
                    print(f"[{utc_now_iso()}] RETOUR À LA NORMALE: {detail}", flush=True)
                    log_action(log_file, "normal", "scale_down_to_baseline", mode, "executed", detail)
                    G_ACTIONS_TOTAL.labels(type="normal", action="scale_down_to_baseline", status="executed").inc()
                else:
                    print(f"[{utc_now_iso()}] Retour à la normale confirmé.", flush=True)

            last_type = current_type

        except Exception as e:
            print(f"[ERREUR] {e}", flush=True)

        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrôleur de remédiation automatique/semi-automatique")
    parser.add_argument("--prom-url", default="http://localhost:9090")
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--mode", default="auto", choices=["auto", "semi-auto"])
    parser.add_argument("--cooldown-seconds", type=int, default=120,
                         help="Délai minimum entre deux actions, pour éviter le sur-déclenchement")
    parser.add_argument("--namespace", default="workload")
    parser.add_argument("--deployment-name", default="sample-workload")
    parser.add_argument("--target-replicas", type=int, default=6,
                         help="Nombre de réplicas nominal (doit correspondre à sample-workload.yaml)")
    parser.add_argument("--scale-up-increment", type=int, default=2)
    parser.add_argument("--max-replicas", type=int, default=10)
    parser.add_argument("--log-file", default="remediation_log.csv")
    parser.add_argument("--pending-file", default="pending_remediation.json")
    parser.add_argument("--metrics-port", type=int, default=9106)
    args = parser.parse_args()

    run(args.prom_url, args.interval_seconds, args.mode, args.cooldown_seconds,
        args.namespace, args.deployment_name, args.target_replicas, args.scale_up_increment,
        args.max_replicas, args.log_file, args.pending_file, args.metrics_port)
