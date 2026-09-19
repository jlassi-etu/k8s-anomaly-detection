#!/usr/bin/env python3
"""
Test rapide et isolé de l'authentification SMTP + envoi d'un email de test,
sans passer par toute la boucle de détection d'anomalie.

Usage:
    export SMTP_PASSWORD='xxxx xxxx xxxx xxxx'
    python3 test_smtp.py --smtp-host smtp.gmail.com --smtp-user vous@gmail.com --email-to vous@gmail.com
"""

import argparse
import os
import smtplib
from email.mime.text import MIMEText

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smtp-host", default="smtp.gmail.com")
    parser.add_argument("--smtp-port", type=int, default=587)
    parser.add_argument("--smtp-user", required=True)
    parser.add_argument("--email-to", required=True)
    args = parser.parse_args()

    password = os.environ.get("SMTP_PASSWORD", "")
    if not password:
        raise SystemExit("Erreur: la variable d'environnement SMTP_PASSWORD est vide.")

    print(f"Longueur du mot de passe lu: {len(password)} caractères")
    print(f"Connexion à {args.smtp_host}:{args.smtp_port} avec l'utilisateur {args.smtp_user}...")

    msg = MIMEText("Ceci est un email de test du projet K8s Anomaly Detection.")
    msg["Subject"] = "Test SMTP - K8s Anomaly Detection"
    msg["From"] = args.smtp_user
    msg["To"] = args.email_to

    try:
        with smtplib.SMTP(args.smtp_host, args.smtp_port, timeout=15) as server:
            server.set_debuglevel(1)  # affiche le dialogue SMTP complet pour diagnostic
            server.starttls()
            server.login(args.smtp_user, password)
            server.sendmail(args.smtp_user, [args.email_to], msg.as_string())
        print("\n✅ SUCCÈS: email de test envoyé.")
    except smtplib.SMTPAuthenticationError as e:
        print(f"\n❌ ÉCHEC D'AUTHENTIFICATION: {e}")
        print("\nVérifiez dans l'ordre:")
        print("  1. La validation en 2 étapes est-elle active sur le compte Google ?")
        print("     -> https://myaccount.google.com/security")
        print("  2. Le mot de passe d'application est-il valide (pas expiré/révoqué) ?")
        print("     -> https://myaccount.google.com/apppasswords")
        print("  3. SMTP_PASSWORD contient-il exactement le mot de passe généré (16 caractères) ?")
    except Exception as e:
        print(f"\n❌ ÉCHEC (autre erreur): {e}")
