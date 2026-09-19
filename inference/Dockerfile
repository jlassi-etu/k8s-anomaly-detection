# Image légère pour l'exporteur d'inférence IA (Isolation Forest + classifieur
# de type + Random Forest de charge). scikit-learn est nécessaire même si le
# script ne l'importe pas directement: joblib.load() a besoin des classes
# sklearn pour désérialiser les modèles (IsolationForest, RandomForestClassifier,
# RandomForestRegressor, StandardScaler).

FROM python:3.12-slim

WORKDIR /app

# IMPORTANT: les versions de joblib/numpy/scikit-learn DOIVENT être compatibles
# avec celles utilisées pour ENTRAÎNER les modèles (sinon joblib.load() peut
# échouer ou donner des résultats incorrects). Avant de builder, vérifiez vos
# versions sur l'hôte avec:
#   pip show scikit-learn numpy joblib
# et épinglez les versions dans requirements.txt si elles diffèrent
# (ex: scikit-learn==1.5.2).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le script d'inférence + le module de notification email
COPY ai_inference_exporter.py .
COPY email_notifier.py .

# Les 3 modèles entraînés (générés par 02_train_isolation_forest.py,
# 05_train_type_classifier.py, 03_train_load_predictor.py)
COPY isolation_forest_model.joblib .
COPY anomaly_type_classifier.joblib .
COPY load_predictor_model.joblib .

EXPOSE 9105

# --prom-url pointe vers le service Prometheus interne au cluster (namespace monitoring)
ENTRYPOINT ["python3", "-u", "ai_inference_exporter.py", \
            "--prom-url", "http://kube-prometheus-kube-prome-prometheus.monitoring.svc.cluster.local:9090"]
