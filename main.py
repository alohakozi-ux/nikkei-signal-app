"""
日経225 売買シグナル監視 メインスクリプト（ポジション管理付き）

新規シグナル: 出来高急増+RSIで検知する。ただし既にポジションを持っている
             （記録済みの）銘柄は無視する（重複無視）。
決済アラート: 保有中の銘柄が-8%の損切りライン、または保有期間(約3週間)に
             達したら通知する。

使い方:
  python main.py
"""

import sys
import time
import pandas as pd
import yfinance as yf

from signal_logic import judge_signal
from notifier import send_email
from positions import (
    load_positions,
    save_positions,
    has_open_position,
    open_position,
    check_exits,
)

TICKERS_CSV = "tickers_nikkei225.csv"

INTERVAL = "5m"
PERIOD = "5d"
RSI_PERIOD = 14
RSI_BUY_TH = 30.0
RSI_SELL_TH = 70.0
VOLUME_WINDOW = 20
VOLUME_SPIKE_TH = 2.0
BATCH_SIZE = 50


def load_tickers(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"code": str})
    df["yf_symbol"] = df["code"] + ".T"
    return df


def fetch_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
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
        result[symbols[0]] = data
    else:
        for sym in symbols:
            if sym in data.columns.get_level_values(0):
                result[sym] = data[sym]
    return result


def main():
    tickers_df = load_tickers(TICKERS_CSV)
    positions_df = load_positions()

    symbols = tickers_df["yf_symbol"].tolist()

    new_signals = []
    latest_prices = {}

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

            code = sym.replace(".T", "")
            latest_prices[code] = float(df["Close"].iloc[-1])

            if has_open_position(positions_df, code):
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
                name_row = tickers_df.loc[tickers_df["code"] == code, "name"]
                name = name_row.values[0] if len(name_row) else code
                signal["code"] = code
                signal["name"] = name
                new_signals.append(signal)
                positions_df = open_position(
                    positions_df, code, name, signal["type"], signal["price"]
                )

        time.sleep(1)

    positions_df, exit_alerts = check_exits(positions_df, latest_prices)

    save_positions(positions_df)

    if not new_signals and not exit_alerts:
        print("シグナルなし、決済なし。")
        return

    body_parts = []

    if new_signals:
        lines = [
            f"[{s['type']}] {s['code']} {s['name']}  価格:{s['price']}円  "
            f"RSI:{s['rsi']}  出来高倍率:{s['volume_ratio']}倍"
            for s in new_signals
        ]
        body_parts.append("■ 新規シグナル\n" + "\n".join(lines))

    if exit_alerts:
        lines = [
            f"[{a['reason']}] {a['code']} {a['name']} ({a['type']})  "
            f"エントリー:{a['entry_price']}円→現在:{a['exit_price']}円  "
            f"リターン:{a['return_pct']:+.2f}%"
            for a in exit_alerts
        ]
        body_parts.append("■ 決済アラート(損切り／保有期間終了)\n" + "\n".join(lines))

    body = "\n\n".join(body_parts)

    subject_parts = []
    if new_signals:
        subject_parts.append(f"新規{len(new_signals)}件")
    if exit_alerts:
        subject_parts.append(f"決済{len(exit_alerts)}件")
    subject = "【株シグナル通知】" + "・".join(subject_parts)

    send_email(subject, body)
    print(f"新規シグナル{len(new_signals)}件、決済アラート{len(exit_alerts)}件")


if __name__ == "__main__":
    main()
