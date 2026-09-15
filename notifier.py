"""
Gmail の SMTP を使ってシグナル通知メールを送信するモジュール。

事前準備:
1. Google アカウントで「2段階認証」を有効にする
2. 「アプリパスワード」を発行する（通常のログインパスワードではなく専用の16桁パスワード）
   https://myaccount.google.com/apppasswords
3. 環境変数に以下を設定する
   GMAIL_ADDRESS   送信元Gmailアドレス
   GMAIL_APP_PASSWORD  発行したアプリパスワード
   NOTIFY_TO       通知を受け取るメールアドレス（複数可、カンマ区切り）
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate


def send_email(subject: str, body: str) -> None:
    """件名と本文を指定して、環境変数の設定に従いGmail経由でメールを送る汎用関数"""
    gmail_address = os.environ.get("GMAIL_ADDRESS")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD")
    notify_to = os.environ.get("NOTIFY_TO")

    if not (gmail_address and gmail_app_password and notify_to):
        raise RuntimeError(
            "環境変数 GMAIL_ADDRESS / GMAIL_APP_PASSWORD / NOTIFY_TO を設定してください"
        )

    to_addrs = [addr.strip() for addr in notify_to.split(",")]

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = ", ".join(to_addrs)
    msg["Date"] = formatdate(localtime=True)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(gmail_address, gmail_app_password)
        server.sendmail(gmail_address, to_addrs, msg.as_string())


def send_signal_email(signals: list[dict]) -> None:
    if not signals:
        return

    lines = []
    for s in signals:
        lines.append(
            f"[{s['type']}] {s['code']} {s['name']}  "
            f"価格:{s['price']}円  RSI:{s['rsi']}  出来高倍率:{s['volume_ratio']}倍"
        )
    body = "本日の売買シグナル一覧\n\n" + "\n".join(lines)
    send_email(f"【株シグナル通知】{len(signals)}件検出", body)
