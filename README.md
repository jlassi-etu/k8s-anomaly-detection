# Prototype de Détection d'Anomalies et de Prédiction de Charge dans un Cluster Kubernetes

Prototype de 4ème année (Cloud & Architecture) : détection en temps réel des anomalies
d'un cluster Kubernetes via Machine Learning (Isolation Forest + classifieur multi-classes
+ Random Forest de prédiction de charge), avec remédiation automatique/semi-automatique
et notifications par email.

## Architecture

**5 types d'anomalies détectées :** `cpu_spike`, `memory_leak`, `resource_saturation`,
`network_anomaly`, `abnormal_drop`.

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| `infra/` | Cluster Kind, workload cible, manifestes Chaos Mesh |
| `collection/` | Collecte des métriques Prometheus + orchestrateur d'injection d'anomalies |
| `ml/` | Pipeline d'entraînement : préparation des features, Isolation Forest, classifieur de type, Random Forest de charge, visualisations |
| `inference/` | Exporteur d'inférence temps réel (pod K8s) + notifications email |
| `remediation/` | Contrôleur de remédiation automatique/semi-automatique (pod K8s) |
| `monitoring/` | Dashboard Grafana + script d'annotations |
| `scripts/` | Utilitaires (test SMTP, lancement local) |
| `docs/` | Guides détaillés par composant |
| `data/`, `logs/` | Données et journaux générés (gitignorés) |

## Démarrage rapide

Guides détaillés dans `docs/` :
- `docs/README-collecte.md` — déployer le cluster et collecter un dataset labellisé
- `docs/README-ml-pipeline.md` — entraîner les 3 modèles et visualiser les résultats
- `docs/README-deploy-ai-pod.md` — containeriser et déployer l'inférence temps réel

### Commandes essentielles (depuis la racine du dépôt)

```bash
# 1. Collecte (5h, avec injection contrôlée des 5 anomalies)
python3 collection/chaos_orchestrator.py --schedule collection/anomaly_schedule.json --base-dir .
python3 collection/collect_metrics.py --duration-hours 5

# 2. Entraînement des modèles
python3 ml/01_prepare_features.py --input data/metrics_dataset_clean.csv --output data/features_dataset.csv
python3 ml/02_train_isolation_forest.py --input data/features_dataset.csv
python3 ml/05_train_type_classifier.py --input data/features_dataset.csv
python3 ml/03_train_load_predictor.py --input data/features_dataset.csv --target-metric cluster_node_cpu_usage_percent

# 3. Déploiement de l'inférence temps réel (copier les modèles entraînés d'abord)
cp isolation_forest_model.joblib anomaly_type_classifier.joblib load_predictor_model.joblib inference/
docker build -t ai-inference-exporter:latest inference/
kind load docker-image ai-inference-exporter:latest --name k8s-anomaly-cluster
kubectl apply -f inference/k8s-ai-inference-secret.yaml -f inference/k8s-ai-inference.yaml

# 4. Déploiement de la remédiation automatique
docker build -t remediation-controller:latest remediation/
kind load docker-image remediation-controller:latest --name k8s-anomaly-cluster
kubectl apply -f remediation/k8s-remediation.yaml

# 5. Dashboard Grafana
# Importer monitoring/grafana-dashboard.json dans Grafana (Dashboards > Import)
```

## Sécurité

Les identifiants (mot de passe SMTP) ne sont **jamais** commités : voir
`inference/k8s-ai-inference-secret.yaml.example` et
`scripts/start_exporter_with_email.sh.example` pour les modèles à copier et
compléter localement (fichiers réels exclus via `.gitignore`).

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci.yml`) vérifie à chaque push/PR :
- la syntaxe de tous les scripts Python
- la validité de tous les manifestes Kubernetes/Chaos Mesh (YAML) et du dashboard Grafana (JSON)
- que les images Docker se construisent correctement

## Auteur

Projet réalisé dans le cadre d'un stage de fin d'études — Cloud & Architecture.
