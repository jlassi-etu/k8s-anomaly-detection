#!/usr/bin/env python3
"""
Fait tourner les 3 modèles entraînés (Isolation Forest, classifieur de type,
Random Forest de charge) en continu sur les métriques Prometheus fraîches, et
republie les résultats comme métriques Prometheus (endpoint /metrics) afin
qu'ils soient visualisables en direct dans Grafana.

Ce script maintient un petit historique en mémoire (fenêtre glissante) par pod
et au niveau cluster, pour reconstruire les mêmes features (moyennes/écarts
glissants, valeurs décalées) que celles utilisées à l'entraînement.

Prérequis:
    - Port-forward Prometheus actif (http://localhost:9090)
    - Les 3 modèles déjà entraînés: isolation_forest_model.joblib,
      anomaly_type_classifier.joblib, load_predictor_model.joblib
    - pip install prometheus_client (en plus des dépendances déjà installées)

Usage:
    python3 ai_inference_exporter.py
    python3 ai_inference_exporter.py --port 9105 --interval-seconds 15
"""

import argparse
import os
import time
from collections import defaultdict, deque

import joblib
import numpy as np
import requests
from prometheus_client import Gauge, start_http_server

import email_notifier

ROLLING_WINDOW = 5  # doit correspondre à la fenêtre utilisée à l'entraînement

POD_METRIC_QUERIES = {
    "pod_cpu_usage_cores": (
        'sum by(pod) (rate(container_cpu_usage_seconds_total'
        '{namespace="workload", container!=""}[1m]))'
    ),
    "pod_memory_working_set_bytes": (
        'sum by(pod) (container_memory_working_set_bytes'
        '{namespace="workload", container!=""})'
    ),
    "pod_restarts_5m": (
        "sum by(pod) (increase(kube_pod_container_status_restarts_total"
        '{namespace="workload"}[5m]))'
    ),
    "pod_ready_status": (
        'sum by(pod) (kube_pod_status_ready{namespace="workload", condition="true"})'
    ),
}

NODE_METRIC_QUERIES = {
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
}

CLUSTER_AGG = {
    "node_cpu_usage_percent": "mean",
    "node_memory_usage_percent": "mean",
    "node_network_receive_bytes": "sum",
    "node_network_transmit_bytes": "sum",
    "node_network_errors": "sum",
}

POD_METRICS_FOR_ROLLING = [
    "pod_cpu_usage_cores",
    "pod_memory_working_set_bytes",
    "pod_restarts_5m",
    "pod_ready_status",
]

CLUSTER_METRICS_FOR_ROLLING = [
    "cluster_node_cpu_usage_percent",
    "cluster_node_memory_usage_percent",
    "cluster_node_network_receive_bytes",
    "cluster_node_network_transmit_bytes",
    "cluster_node_network_errors",
    "total_pod_cpu_usage_cores",
    "total_pod_memory_working_set_bytes",
]

ALL_LABELS = ["normal", "cpu_spike", "memory_leak", "resource_saturation",
              "network_anomaly", "abnormal_drop"]

# --- Métriques Prometheus exposées ---
G_POD_SCORE = Gauge("k8s_ai_pod_anomaly_score", "Score Isolation Forest par pod (bas = anormal)", ["pod"])
G_POD_IS_ANOMALY = Gauge("k8s_ai_pod_is_anomaly", "1 si le pod est jugé anormal par l'Isolation Forest", ["pod"])
G_CLUSTER_SCORE = Gauge("k8s_ai_cluster_anomaly_score", "Score minimal (pire pod) du cluster")
G_CLUSTER_IS_ANOMALY = Gauge("k8s_ai_cluster_is_anomaly", "1 si au moins un pod est jugé anormal")
G_TYPE_ACTIVE = Gauge("k8s_ai_anomaly_type_active", "1 pour le type d'anomalie actuellement détecté, 0 sinon", ["type"])
G_TYPE_PROBA = Gauge("k8s_ai_anomaly_type_probability", "Probabilité prédite par type d'anomalie", ["type"])
G_PREDICTED_LOAD = Gauge("k8s_ai_predicted_cpu_percent", "CPU cluster prédit (horizon défini au training)")
G_EXPORTER_UP = Gauge("k8s_ai_exporter_up", "1 si le dernier cycle d'inférence s'est déroulé sans erreur")


def query_prometheus(prom_url, promql):
    resp = requests.get(f"{prom_url}/api/v1/query", params={"query": promql}, timeout=10)
    resp.raise_for_status()
    result = resp.json()
    if result.get("status") != "success":
        return {}
    out = {}
    for series in result["data"]["result"]:
        labels = series["metric"]
        key = labels.get("pod") or labels.get("instance") or "_"
        out[key] = float(series["value"][1])
    return out


def build_feature_vector(feature_dict, feature_cols):
    """Aligne un dict {nom_feature: valeur} sur la liste ordonnée attendue par
    un modèle, en mettant 0 pour toute feature manquante."""
    return np.array([[feature_dict.get(col, 0.0) for col in feature_cols]])


def update_rolling(history, key, metric_name, value):
    buf = history[(key, metric_name)]
    buf.append(value)
    values = list(buf)
    mean = float(np.mean(values))
    std = float(np.std(values)) if len(values) > 1 else 0.0
    lag1 = values[-2] if len(values) > 1 else value
    return mean, std, lag1


def run(prom_url, port, interval_seconds, if_model_path, type_model_path, load_model_path, debounce_cycles, debug,
        smtp_config=None, grafana_url=None, recovery_cycles=4, email_reminder_seconds=900):
    print("Chargement des modèles...")
    if_bundle = joblib.load(if_model_path)
    type_bundle = joblib.load(type_model_path)
    load_bundle = joblib.load(load_model_path)

    if_model, if_cols = if_bundle["model"], if_bundle["feature_cols"]
    if_scaler = if_bundle.get("scaler")  # StandardScaler utilisé à l'entraînement (02_train_isolation_forest.py)
    type_model, type_cols, type_classes = type_bundle["model"], type_bundle["feature_cols"], type_bundle["classes"]
    load_model, load_cols = load_bundle["model"], load_bundle["feature_cols"]

    print(f"  Isolation Forest: {len(if_cols)} features, scaler={'chargé' if if_scaler is not None else 'ABSENT (risque de scores faux!)'}")
    print(f"  Classifieur de type: {len(type_cols)} features, classes={type_classes}")
    print(f"  Prédicteur de charge: {len(load_cols)} features, cible={load_bundle.get('target_metric')}")

    start_http_server(port)
    print(f"\nExporteur Prometheus démarré sur http://0.0.0.0:{port}/metrics")
    print(f"Boucle d'inférence toutes les {interval_seconds}s. Ctrl+C pour arrêter.\n")

    pod_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    cluster_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    consecutive_anomaly_cycles = 0
    consecutive_normal_cycles = 0
    confirmed_state = "normal"
    last_emailed_state = "normal"
    episode_start_time = 0.0
    last_reminder_time = 0.0

    while True:
        try:
            # --- 1. Récupérer le snapshot courant ---
            pod_raw = {m: query_prometheus(prom_url, q) for m, q in POD_METRIC_QUERIES.items()}
            node_raw = {m: query_prometheus(prom_url, q) for m, q in NODE_METRIC_QUERIES.items()}

            pods = set()
            for m in pod_raw.values():
                pods.update(m.keys())

            # --- 2. Agrégats cluster ---
            cluster_raw = {}
            for metric, agg in CLUSTER_AGG.items():
                vals = list(node_raw.get(metric, {}).values())
                if vals:
                    cluster_raw[f"cluster_{metric}"] = float(np.mean(vals)) if agg == "mean" else float(np.sum(vals))
                else:
                    cluster_raw[f"cluster_{metric}"] = 0.0

            cluster_raw["total_pod_cpu_usage_cores"] = sum(pod_raw["pod_cpu_usage_cores"].values())
            cluster_raw["total_pod_memory_working_set_bytes"] = sum(pod_raw["pod_memory_working_set_bytes"].values())

            # --- 3. Features cluster (pour le prédicteur de charge) ---
            cluster_features = dict(cluster_raw)
            for metric in CLUSTER_METRICS_FOR_ROLLING:
                value = cluster_raw.get(metric, 0.0)
                mean, std, lag1 = update_rolling(cluster_history, "_cluster", metric, value)
                cluster_features[f"{metric}_lag1"] = lag1
                for lag in range(2, ROLLING_WINDOW + 1):
                    cluster_features.setdefault(f"{metric}_lag{lag}", lag1)  # approximation si historique court
                cluster_features[f"{metric}_roll_mean"] = mean

            X_load = build_feature_vector(cluster_features, load_cols)
            predicted_load = float(load_model.predict(X_load)[0])
            G_PREDICTED_LOAD.set(predicted_load)

            # --- 4. Features + prédiction par pod ---
            worst_score = None
            worst_pod_features = None
            worst_pod_name = None
            any_anomaly = False

            for pod in pods:
                pod_features = dict(cluster_raw)
                for metric in POD_METRICS_FOR_ROLLING:
                    value = pod_raw.get(metric, {}).get(pod, 0.0)
                    pod_features[metric] = value
                    mean, std, lag1 = update_rolling(pod_history, pod, metric, value)
                    pod_features[f"{metric}_roll_mean"] = mean
                    pod_features[f"{metric}_roll_std"] = std
                    pod_features[f"{metric}_lag1"] = lag1

                X_if = build_feature_vector(pod_features, if_cols)
                if if_scaler is not None:
                    X_if = if_scaler.transform(X_if)
                score = float(if_model.decision_function(X_if)[0])
                is_anomaly = bool(if_model.predict(X_if)[0] == -1)

                G_POD_SCORE.labels(pod=pod).set(score)
                G_POD_IS_ANOMALY.labels(pod=pod).set(1 if is_anomaly else 0)

                if is_anomaly:
                    any_anomaly = True

                if worst_score is None or score < worst_score:
                    worst_score = score
                    worst_pod_features = pod_features
                    worst_pod_name = pod

            # --- 5. Score/état cluster + type d'anomalie (basé sur le pod le pire) ---
            if worst_score is not None:
                G_CLUSTER_SCORE.set(worst_score)

            predicted_type = "normal"
            probabilities = {label: 0.0 for label in ALL_LABELS}

            if worst_pod_features is not None:
                X_type = build_feature_vector(worst_pod_features, type_cols)
                proba = type_model.predict_proba(X_type)[0]
                for cls, p in zip(type_classes, proba):
                    probabilities[cls] = float(p)

                if any_anomaly:
                    # parmi les classes différentes de "normal", on prend la plus probable
                    non_normal = {k: v for k, v in probabilities.items() if k != "normal"}
                    predicted_type = max(non_normal, key=non_normal.get)
                else:
                    predicted_type = "normal"

            # --- Hystérésis SYMÉTRIQUE: il faut plusieurs cycles consécutifs
            # anormaux pour ENTRER en anomalie, ET plusieurs cycles consécutifs
            # normaux pour EN SORTIR. Sans ça, un score qui oscille juste autour
            # de zéro fait clignoter l'état à chaque cycle (observé en pratique).
            if any_anomaly:
                consecutive_anomaly_cycles += 1
                consecutive_normal_cycles = 0
            else:
                consecutive_normal_cycles += 1
                consecutive_anomaly_cycles = 0

            if confirmed_state == "normal":
                if consecutive_anomaly_cycles >= debounce_cycles:
                    confirmed_state = predicted_type
            else:
                if consecutive_normal_cycles >= recovery_cycles:
                    confirmed_state = "normal"
                elif any_anomaly and consecutive_anomaly_cycles >= debounce_cycles:
                    # Anomalie toujours active mais le type dominant a changé
                    # (ex: cpu_spike qui évolue vers resource_saturation).
                    confirmed_state = predicted_type

            for label in ALL_LABELS:
                G_TYPE_ACTIVE.labels(type=label).set(1 if label == confirmed_state else 0)
                G_TYPE_PROBA.labels(type=label).set(probabilities[label])

            G_CLUSTER_IS_ANOMALY.set(1 if confirmed_state != "normal" else 0)

            G_EXPORTER_UP.set(1)
            status = f"ANOMALIE={confirmed_state}" if confirmed_state != "normal" else "normal"
            raw_hint = f" (brut: {predicted_type}, {consecutive_anomaly_cycles}/{debounce_cycles} cycles)" if any_anomaly else ""
            print(f"[{time.strftime('%H:%M:%S')}] {len(pods)} pods | "
                  f"score cluster={worst_score:.3f} | {status}{raw_hint} | "
                  f"charge prédite={predicted_load:.1f}%", flush=True)

            # --- Machine à états pour l'email: DÉTECTÉE (une fois au début de
            # l'épisode) -> RAPPEL (toutes les email_reminder_seconds tant que
            # ça persiste) -> RÉSOLUE (une fois au retour au calme confirmé).
            if smtp_config is not None:
                details = {
                    "anomaly_type": confirmed_state,
                    "worst_pod": worst_pod_name or "inconnu",
                    "if_score": worst_score if worst_score is not None else 0.0,
                    "predicted_load_percent": predicted_load,
                    "probabilities": probabilities,
                    "cluster_cpu_percent": cluster_raw.get("cluster_node_cpu_usage_percent", 0.0),
                    "cluster_memory_percent": cluster_raw.get("cluster_node_memory_usage_percent", 0.0),
                    "cluster_network_receive_bytes": cluster_raw.get("cluster_node_network_receive_bytes", 0.0),
                    "cluster_network_transmit_bytes": cluster_raw.get("cluster_node_network_transmit_bytes", 0.0),
                    "cluster_network_errors": cluster_raw.get("cluster_node_network_errors", 0.0),
                    "total_pod_cpu_cores": cluster_raw.get("total_pod_cpu_usage_cores", 0.0),
                    "total_pod_memory_bytes": cluster_raw.get("total_pod_memory_working_set_bytes", 0.0),
                    "n_pods": len(pods),
                }
                now = time.time()

                if confirmed_state != "normal":
                    if last_emailed_state == "normal":
                        # Nouvelle anomalie: email "détectée"
                        try:
                            subject = email_notifier.notify_anomaly(smtp_config, details, grafana_url, kind="detected")
                            print(f"  [EMAIL] Envoyé (détectée): {subject}", flush=True)
                        except Exception as e:
                            print(f"  [EMAIL] Échec de l'envoi: {e}", flush=True)
                        episode_start_time = now
                        last_reminder_time = now
                        last_emailed_state = confirmed_state
                    elif now - last_reminder_time >= email_reminder_seconds:
                        # Toujours en cours après le délai de rappel: email "rappel"
                        details["duration_minutes"] = (now - episode_start_time) / 60.0
                        try:
                            subject = email_notifier.notify_anomaly(smtp_config, details, grafana_url, kind="reminder")
                            print(f"  [EMAIL] Envoyé (rappel): {subject}", flush=True)
                        except Exception as e:
                            print(f"  [EMAIL] Échec de l'envoi: {e}", flush=True)
                        last_reminder_time = now
                        last_emailed_state = confirmed_state
                    else:
                        last_emailed_state = confirmed_state
                else:
                    if last_emailed_state != "normal":
                        # Retour à la normale confirmé: email "résolue"
                        details["anomaly_type"] = last_emailed_state
                        details["duration_minutes"] = (now - episode_start_time) / 60.0
                        try:
                            subject = email_notifier.notify_anomaly(smtp_config, details, grafana_url, kind="resolved")
                            print(f"  [EMAIL] Envoyé (résolue): {subject}", flush=True)
                        except Exception as e:
                            print(f"  [EMAIL] Échec de l'envoi: {e}", flush=True)
                    last_emailed_state = "normal"

            if debug and worst_pod_features is not None:
                import json
                print("  [DEBUG] Features envoyées au modèle (pod le pire), dans l'ordre attendu:")
                ordered = {col: worst_pod_features.get(col, 0.0) for col in if_cols}
                print("  " + json.dumps(ordered, indent=2), flush=True)

        except Exception as e:
            G_EXPORTER_UP.set(0)
            print(f"[ERREUR] {e}", flush=True)

        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporteur d'inférence IA temps réel pour Grafana")
    parser.add_argument("--prom-url", default="http://localhost:9090")
    parser.add_argument("--port", type=int, default=9105)
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--if-model", default="isolation_forest_model.joblib")
    parser.add_argument("--type-model", default="anomaly_type_classifier.joblib")
    parser.add_argument("--load-model", default="load_predictor_model.joblib")
    parser.add_argument(
        "--debounce-cycles",
        type=int,
        default=2,
        help="Nombre de cycles consécutifs d'anomalie requis avant de la confirmer "
             "(réduit le scintillement dû au bruit statistique, défaut: 2 cycles = ~30s)",
    )
    parser.add_argument(
        "--recovery-cycles",
        type=int,
        default=int(os.environ.get("RECOVERY_CYCLES", "4")),
        help="Nombre de cycles consécutifs NORMAUX requis pour sortir d'une anomalie "
             "confirmée (évite le clignotement si le score oscille près de zéro, "
             "défaut: 4 cycles = ~60s)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Affiche le détail des features envoyées au modèle à chaque cycle",
    )

    email_group = parser.add_argument_group("Notification par email (optionnel)")
    email_group.add_argument(
        "--smtp-host", default=os.environ.get("SMTP_HOST"),
        help="ex: smtp.gmail.com (ou variable d'env SMTP_HOST)",
    )
    email_group.add_argument(
        "--smtp-port", type=int, default=int(os.environ.get("SMTP_PORT", "587")),
    )
    email_group.add_argument(
        "--smtp-user", default=os.environ.get("SMTP_USER"),
        help="ex: votre.compte@gmail.com (ou variable d'env SMTP_USER)",
    )
    email_group.add_argument(
        "--smtp-no-tls", action="store_true",
        help="Désactive STARTTLS (à éviter, seulement pour un relais SMTP local non sécurisé)",
    )
    email_group.add_argument(
        "--email-from", default=os.environ.get("EMAIL_FROM"),
        help="Défaut: identique à --smtp-user (ou variable d'env EMAIL_FROM)",
    )
    email_group.add_argument(
        "--email-to", default=os.environ.get("EMAIL_TO"),
        help="Un ou plusieurs destinataires séparés par des virgules (ou variable d'env EMAIL_TO)",
    )
    email_group.add_argument(
        "--grafana-url", default=os.environ.get("GRAFANA_URL"),
        help="URL du dashboard Grafana à inclure dans l'email (optionnel, ou variable d'env GRAFANA_URL)",
    )
    email_group.add_argument(
        "--email-reminder-minutes",
        type=float,
        default=float(os.environ.get("EMAIL_REMINDER_MINUTES", "15")),
        help="Délai avant un email de rappel si l'anomalie persiste (défaut: 15 min)",
    )
    args = parser.parse_args()

    smtp_config = None
    if args.smtp_host and args.email_to:
        smtp_password = os.environ.get("SMTP_PASSWORD", "")
        if args.smtp_user and not smtp_password:
            raise SystemExit(
                "Erreur: --smtp-user est fourni mais la variable d'environnement "
                "SMTP_PASSWORD est vide. Définissez-la avant de lancer le script:\n"
                "  export SMTP_PASSWORD='votre_mot_de_passe_ou_app_password'\n"
                "(Ne passez JAMAIS le mot de passe en argument --xxx, il serait visible "
                "de tous via `ps aux`.)"
            )
        smtp_config = {
            "host": args.smtp_host,
            "port": args.smtp_port,
            "user": args.smtp_user,
            "password": smtp_password,
            "email_from": args.email_from or args.smtp_user,
            "email_to": [addr.strip() for addr in args.email_to.split(",") if addr.strip()],
            "use_tls": not args.smtp_no_tls,
        }
        print(f"Notification email activée -> {', '.join(smtp_config['email_to'])} "
              f"(rappel toutes les {args.email_reminder_minutes:.0f} min si persistant)")

    run(args.prom_url, args.port, args.interval_seconds, args.if_model, args.type_model,
        args.load_model, args.debounce_cycles, args.debug, smtp_config, args.grafana_url,
        args.recovery_cycles, args.email_reminder_minutes * 60)
