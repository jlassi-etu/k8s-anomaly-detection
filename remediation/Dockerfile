# Image pour le contrôleur de remédiation automatique/semi-automatique.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY remediation_controller.py .
COPY approve_remediation.py .

EXPOSE 9106

# --prom-url pointe vers le service Prometheus interne au cluster
ENTRYPOINT ["python3", "-u", "remediation_controller.py", \
            "--prom-url", "http://kube-prometheus-kube-prome-prometheus.monitoring.svc.cluster.local:9090", \
            "--mode", "auto"]
