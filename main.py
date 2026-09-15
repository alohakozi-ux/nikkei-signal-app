"""
日経225 売買シグナル監視 メインスクリプト

使い方:
  python main.py

想定運用:
  クラウド(cron / Cloud Scheduler / GitHub Actions 等)から
  場中に数分間隔（例: 5分おき）で本スクリプトを実行する。
  条件に合致した銘柄があればメールで通知する。
"""

import sys
import time
import pandas as pd
import yfinance as yf

from signal_logic import judge_signal
from notifier import send_signal_email

TICKERS_CSV = "tickers_nikkei225.csv"

# --- パラメータ（必要に応じて調整） ---
INTERVAL = "5m"       # 足の間隔: 1m/5m/15m/1d など
PERIOD = "5d"         # 取得期間
RSI_PERIOD = 14
RSI_BUY_TH = 30.0
RSI_SELL_TH = 70.0
VOLUME_WINDOW = 20
VOLUME_SPIKE_TH = 2.0
BATCH_SIZE = 50       # yfinanceに一度に渡す銘柄数（多すぎるとエラーになりやすいため分割）


def load_tickers(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"code": str})
    df["yf_symbol"] = df["code"] + ".T"
    return df


def fetch_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """yfinanceで複数銘柄をまとめて取得し、銘柄ごとのDataFrameに分割する"""
    data = yf.download(
        tickers=" ".join(symbols),
        period=PERIOD,
        interval=INTERVAL,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    result = {}
    if len(symbols) == 1:
        # 単一銘柄の場合は列構造が異なるため個別対応
        result[symbols[0]] = data
    else:
        for sym in symbols:
            if sym in data.columns.get_level_values(0):
                result[sym] = data[sym]
    return result


def main():
    tickers_df = load_tickers(TICKERS_CSV)
    all_signals = []

    symbols = tickers_df["yf_symbol"].tolist()

    for i in range(0, len(symbols), BATCH_SIZE):
        batch_symbols = symbols[i : i + BATCH_SIZE]
        try:
            batch_data = fetch_batch(batch_symbols)
        except Exception as e:
            print(f"取得エラー(バッチ {i}): {e}", file=sys.stderr)
            continue

        for sym, df in batch_data.items():
            df = df.dropna(how="all")
            if df.empty or "Close" not in df.columns:
                continue

            signal = judge_signal(
                df,
                rsi_period=RSI_PERIOD,
                rsi_buy_th=RSI_BUY_TH,
                rsi_sell_th=RSI_SELL_TH,
                volume_window=VOLUME_WINDOW,
                volume_spike_th=VOLUME_SPIKE_TH,
            )
            if signal:
                code = sym.replace(".T", "")
                name_row = tickers_df.loc[tickers_df["code"] == code, "name"]
                name = name_row.values[0] if len(name_row) else code
                signal["code"] = code
                signal["name"] = name
                all_signals.append(signal)

        # yfinanceへの過度なリクエストを避けるための小休止
        time.sleep(1)

    if all_signals:
        print(f"{len(all_signals)}件のシグナルを検出。メール送信します。")
        send_signal_email(all_signals)
    else:
        print("シグナルなし。")


if __name__ == "__main__":
    main()
