#!/usr/bin/env python3
"""
Construit et envoie un email détaillé lors d'une anomalie confirmée.
Utilisé par ai_inference_exporter.py.
"""

import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

ANOMALY_LABELS_FR = {
    "cpu_spike": "Pic anormal de CPU",
    "memory_leak": "Fuite mémoire",
    "resource_saturation": "Saturation des ressources",
    "network_anomaly": "Anomalie réseau",
    "abnormal_drop": "Baisse anormale de disponibilité",
}


def _fmt_bytes(n):
    for unit in ["o", "Ko", "Mo", "Go"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} To"


def build_email_content(details, grafana_url=None, kind="detected"):
    """
    kind: "detected" (nouvelle anomalie), "reminder" (persiste encore après le
    délai configuré), ou "resolved" (retour à la normale confirmé).
    details attend en plus, pour reminder/resolved: 'duration_minutes'.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    label_fr = ANOMALY_LABELS_FR.get(details["anomaly_type"], details["anomaly_type"])

    if kind == "resolved":
        duration = details.get("duration_minutes", 0.0)
        subject = f"[K8s Anomaly Detection] RÉSOLU: {label_fr} (durée: {duration:.1f} min)"
        header = f"✅ ANOMALIE RÉSOLUE — {label_fr}"
        header_color = "#27ae60"
        intro = f"Le cluster est revenu à un état normal après {duration:.1f} minutes d'anomalie.\n"
        intro_html = f"<p>Le cluster est revenu à un état normal après <b>{duration:.1f} minutes</b> d'anomalie.</p>"
    elif kind == "reminder":
        duration = details.get("duration_minutes", 0.0)
        subject = f"[K8s Anomaly Detection] TOUJOURS EN COURS: {label_fr} (depuis {duration:.0f} min)"
        header = f"⏰ ANOMALIE TOUJOURS EN COURS — {label_fr}"
        header_color = "#e67e22"
        intro = f"Cette anomalie dure depuis {duration:.0f} minutes sans résolution.\n"
        intro_html = f"<p>Cette anomalie dure depuis <b>{duration:.0f} minutes</b> sans résolution.</p>"
    else:
        subject = f"[K8s Anomaly Detection] Anomalie détectée: {label_fr} ({ts})"
        header = f"🔴 ANOMALIE DÉTECTÉE — {label_fr}"
        header_color = "#c0392b"
        intro = ""
        intro_html = ""

    proba_lines = "\n".join(
        f"  - {t:<22} {p*100:5.1f}%" for t, p in sorted(details["probabilities"].items(), key=lambda x: -x[1])
    )
    proba_rows_html = "".join(
        f"<tr><td>{t}</td><td style='text-align:right'>{p*100:.1f}%</td></tr>"
        for t, p in sorted(details["probabilities"].items(), key=lambda x: -x[1])
    )

    text_body = f"""{header}

{intro}Horodatage (UTC) : {ts}
Type             : {details['anomaly_type']} ({label_fr})
Pod le plus affecté : {details['worst_pod']}
Score Isolation Forest : {details['if_score']:.4f} (négatif = anormal)

Probabilités par type (classifieur) :
{proba_lines}

--- Métriques cluster au moment de la détection ---
CPU moyen des nœuds      : {details['cluster_cpu_percent']:.1f} %
Mémoire moyenne des nœuds: {details['cluster_memory_percent']:.1f} %
Réseau reçu (cluster)    : {_fmt_bytes(details['cluster_network_receive_bytes'])}/s
Réseau émis (cluster)    : {_fmt_bytes(details['cluster_network_transmit_bytes'])}/s
Erreurs réseau           : {details['cluster_network_errors']:.2f}/s
CPU total pods workload  : {details['total_pod_cpu_cores']:.3f} cores
Mémoire totale pods      : {_fmt_bytes(details['total_pod_memory_bytes'])}
Nombre de pods surveillés: {details['n_pods']}

--- Prédiction ---
Charge CPU prédite (horizon d'entraînement) : {details['predicted_load_percent']:.1f} %
"""

    if grafana_url:
        text_body += f"\nDashboard Grafana : {grafana_url}\n"

    html_body = f"""
    <html><body style="font-family: Arial, sans-serif; color: #222;">
    <h2 style="color:{header_color};">{header}</h2>
    {intro_html}
    <p><b>Horodatage (UTC) :</b> {ts}<br>
       <b>Type :</b> {details['anomaly_type']} ({label_fr})<br>
       <b>Pod le plus affecté :</b> {details['worst_pod']}<br>
       <b>Score Isolation Forest :</b> {details['if_score']:.4f} (négatif = anormal)</p>

    <h3>Probabilités par type</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
      <tr style="background:#eee;"><th>Type</th><th>Probabilité</th></tr>
      {proba_rows_html}
    </table>

    <h3>Métriques cluster au moment de la détection</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
      <tr><td>CPU moyen des nœuds</td><td>{details['cluster_cpu_percent']:.1f} %</td></tr>
      <tr><td>Mémoire moyenne des nœuds</td><td>{details['cluster_memory_percent']:.1f} %</td></tr>
      <tr><td>Réseau reçu (cluster)</td><td>{_fmt_bytes(details['cluster_network_receive_bytes'])}/s</td></tr>
      <tr><td>Réseau émis (cluster)</td><td>{_fmt_bytes(details['cluster_network_transmit_bytes'])}/s</td></tr>
      <tr><td>Erreurs réseau</td><td>{details['cluster_network_errors']:.2f}/s</td></tr>
      <tr><td>CPU total pods workload</td><td>{details['total_pod_cpu_cores']:.3f} cores</td></tr>
      <tr><td>Mémoire totale pods</td><td>{_fmt_bytes(details['total_pod_memory_bytes'])}</td></tr>
      <tr><td>Nombre de pods surveillés</td><td>{details['n_pods']}</td></tr>
    </table>

    <h3>Prédiction</h3>
    <p>Charge CPU prédite : <b>{details['predicted_load_percent']:.1f} %</b></p>
    """
    if grafana_url:
        html_body += f'<p><a href="{grafana_url}">Voir le dashboard Grafana</a></p>'
    html_body += "</body></html>"

    return subject, text_body, html_body


def send_email(smtp_host, smtp_port, smtp_user, smtp_password, email_from, email_to,
                subject, text_body, html_body, use_tls=True):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = ", ".join(email_to)
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
        if use_tls:
            server.starttls()
        if smtp_user:
            server.login(smtp_user, smtp_password)
        server.sendmail(email_from, email_to, msg.as_string())


def notify_anomaly(smtp_config, details, grafana_url=None, kind="detected"):
    """Construit et envoie l'email. smtp_config est un dict avec host, port,
    user, password, email_from, email_to (liste), use_tls."""
    subject, text_body, html_body = build_email_content(details, grafana_url, kind)
    send_email(
        smtp_config["host"], smtp_config["port"], smtp_config["user"], smtp_config["password"],
        smtp_config["email_from"], smtp_config["email_to"], subject, text_body, html_body,
        smtp_config.get("use_tls", True),
    )
    return subject
