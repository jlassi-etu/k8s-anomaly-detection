#!/bin/bash
# Réorganise le projet en dossiers logiques et corrige les chemins internes
# qui en dépendent (anomaly_schedule.json, .gitignore).
#
# À lancer UNE SEULE FOIS depuis la racine de ~/k8s-anomaly-project.
# Sûr à relancer plusieurs fois (idempotent : ignore ce qui est déjà déplacé).

set -e
cd "$(dirname "$0")"

echo "=== Création de la structure de dossiers ==="
mkdir -p infra/chaos-manifests
mkdir -p collection
mkdir -p ml
mkdir -p inference
mkdir -p remediation
mkdir -p monitoring
mkdir -p scripts
mkdir -p docs
mkdir -p data
mkdir -p logs
mkdir -p .github/workflows

move() {
    # move <source> <destination> -- ignore silencieusement si la source n'existe pas
    if [ -e "$1" ]; then
        mv -v "$1" "$2"
    fi
}

echo "=== Infrastructure (manifestes K8s / Chaos Mesh) ==="
move sample-workload.yaml infra/sample-workload.yaml
if [ -d chaos-manifests ] && [ ! -d infra/chaos-manifests/01-cpu-spike.yaml ]; then
    mv -v chaos-manifests/*.yaml infra/chaos-manifests/ 2>/dev/null || true
    rmdir chaos-manifests 2>/dev/null || true
fi

echo "=== Collecte de données ==="
move collect_metrics.py collection/collect_metrics.py
move chaos_orchestrator.py collection/chaos_orchestrator.py
move anomaly_schedule.json collection/anomaly_schedule.json
# Les manifestes chaos ont déménagé vers infra/chaos-manifests/ : on corrige
# les chemins référencés dans le planning (invocation prévue depuis la racine
# du dépôt, voir docs/COMMANDS.md).
if [ -f collection/anomaly_schedule.json ]; then
    sed -i 's#"manifest": "chaos-manifests/#"manifest": "infra/chaos-manifests/#g' collection/anomaly_schedule.json
fi

echo "=== Pipeline Machine Learning ==="
move 01_prepare_features.py ml/01_prepare_features.py
move 02_train_isolation_forest.py ml/02_train_isolation_forest.py
move 03_train_load_predictor.py ml/03_train_load_predictor.py
move 04_visualize_dataset.py ml/04_visualize_dataset.py
move 05_train_type_classifier.py ml/05_train_type_classifier.py
move feature_ranges_reference.py ml/feature_ranges_reference.py

echo "=== Inférence temps réel (contexte Docker autonome) ==="
move Dockerfile.ai-inference inference/Dockerfile
move requirements-ai-inference.txt inference/requirements.txt
move ai_inference_exporter.py inference/ai_inference_exporter.py
move email_notifier.py inference/email_notifier.py
move k8s-ai-inference.yaml inference/k8s-ai-inference.yaml
move k8s-ai-inference-secret.yaml inference/k8s-ai-inference-secret.yaml
move k8s-ai-inference-secret.yaml.example inference/k8s-ai-inference-secret.yaml.example
# Modèles entraînés: déplacés (regénérables via ml/02, ml/03, ml/05 --
# après un réentraînement futur, recopiez le nouveau .joblib dans inference/
# avant de rebuilder l'image Docker)
for f in isolation_forest_model.joblib anomaly_type_classifier.joblib load_predictor_model.joblib; do
    move "$f" "inference/$f"
done

echo "=== Remédiation automatique (contexte Docker autonome) ==="
move Dockerfile.remediation remediation/Dockerfile
move requirements-remediation.txt remediation/requirements.txt
move remediation_controller.py remediation/remediation_controller.py
move approve_remediation.py remediation/approve_remediation.py
move k8s-remediation.yaml remediation/k8s-remediation.yaml

echo "=== Monitoring / Grafana ==="
# Consolide les éventuels doublons "grafana-dashboard(2).json" etc.
for f in grafana-dashboard\(*\).json; do
    [ -e "$f" ] && rm -v "$f"
done
move grafana-dashboard.json monitoring/grafana-dashboard.json
move push_annotations_to_grafana.py monitoring/push_annotations_to_grafana.py

echo "=== Scripts utilitaires ==="
move test_smtp.py scripts/test_smtp.py
move start_exporter_with_email.sh scripts/start_exporter_with_email.sh
move start_exporter_with_email.sh.example scripts/start_exporter_with_email.sh.example

echo "=== Documentation ==="
move README-collecte.md docs/README-collecte.md
move README-ml-pipeline.md docs/README-ml-pipeline.md
move README-deploy-ai-pod.md docs/README-deploy-ai-pod.md

echo "=== Données générées / logs (gitignorés, pour garder la racine propre) ==="
for f in metrics_dataset*.csv features_dataset*.csv anomaly_events.csv \
         anomaly_state.json load_predictions.csv remediation_log.csv \
         pending_remediation.json; do
    move "$f" "data/$f"
done
for f in *.log; do
    move "$f" "logs/$f"
done
move nohup.out logs/nohup.out

echo "=== Nettoyage ==="
rm -rf __pycache__ ml/__pycache__ inference/__pycache__ remediation/__pycache__ collection/__pycache__
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "(venv/ et figures/ laissés tels quels — gitignorés, pas besoin de les déplacer)"

echo ""
echo "=== Terminé ==="
echo "Nouvelle structure:"
find . -maxdepth 2 -not -path "./venv*" -not -path "./.git*" -not -path "./figures*" | sort
