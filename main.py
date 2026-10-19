"""
日経225 買いシグナル通知 メインスクリプト (日足・1日1回、大引け後に実行)

新規シグナル: 日足で「RSI25以下」かつ「出来高が直近平均の2倍以上」の銘柄を「買い」として通知する。
             (過去約4年のバックテストで、この条件が最も安定していた)
             既に買いのポジションを持っている(記録済みの)銘柄は無視する(重複無視)。
決済アラート: 保有中の銘柄が-8%の損切りライン、+10%の利確ライン、
             または保有期間(約3週間)に達したら通知する。

使い方:
  python main.py            # 通常実行(休場日は何もしない)
  python main.py --force    # 休場日でも実行する(テスト用)
"""

import sys
import time
from dataclasses import replace

import pandas as pd
import yfinance as yf

from notifier import send_email
from positions import (
    load_positions,
    save_positions,
    has_open_position,
    open_position,
    check_exits,
    today_jst,
)
from signal_logic_v2 import OLD_CONFIG, judge_signal

TICKERS_CSV = "tickers_nikkei225.csv"

PERIOD = "6mo"
BATCH_SIZE = 50

# 買いのみ・RSI25以下・出来高2倍以上。数字を変えたいときはここを書き換える
# (例: RSI20にすると、通知は減るが、バックテストでは1件あたりの成績が良かった)
CONFIG = replace(OLD_CONFIG, rsi_buy_th=25.0, volume_spike_th=2.0)


def load_tickers(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"code": str})
    df["yf_symbol"] = df["code"] + ".T"
    return df


def fetch_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=" ".join(symbols),
        period=PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,  # 株式分割を調整した株価(分割で偽のシグナルが出るのを防ぐ)
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
    force = "--force" in sys.argv
    today = today_jst()

    tickers_df = load_tickers(TICKERS_CSV)
    positions_df = load_positions()

    symbols = tickers_df["yf_symbol"].tolist()

    new_signals = []
    latest_prices = {}
    latest_dates = []

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
            latest_dates.append(pd.Timestamp(df.index[-1]).date())

            # 既に買いのポジションがある銘柄は、新しいシグナルを出さない(重複無視)
            if has_open_position(positions_df, code, "BUY"):
                continue

            signal = judge_signal(df, CONFIG)
            if signal and signal["type"] == "BUY":
                name_row = tickers_df.loc[tickers_df["code"] == code, "name"]
                name = name_row.values[0] if len(name_row) else code
                signal["code"] = code
                signal["name"] = name
                new_signals.append(signal)

        time.sleep(1)

    if not latest_dates:
        print("株価を取得できませんでした。", file=sys.stderr)
        return

    # 休場日(土日祝)は、最新の株価が今日のものではないので何もしない
    last_trading_day = max(latest_dates)
    if last_trading_day != today and not force:
        print(f"本日({today})は取引がありません(最新の株価は{last_trading_day})。何もしません。")
        return

    for s in new_signals:
        positions_df = open_position(positions_df, s["code"], s["name"], "BUY", s["price"])

    positions_df, exit_alerts = check_exits(positions_df, latest_prices)

    save_positions(positions_df)

    if not new_signals and not exit_alerts:
        print("シグナルなし、決済なし。")
        return

    body_parts = []

    if new_signals:
        lines = [
            f"[BUY] {s['code']} {s['name']}  終値:{s['price']}円  "
            f"RSI:{s['rsi']}  出来高倍率:{s['volume_ratio']}倍"
            for s in new_signals
        ]
        body_parts.append(
            "■ 新規の買いシグナル(翌朝の始値で買う想定)\n" + "\n".join(lines)
        )

    if exit_alerts:
        lines = []
        for a in exit_alerts:
            ret = "-" if a["return_pct"] is None else f"{a['return_pct']:+.2f}%"
            lines.append(
                f"[{a['reason']}] {a['code']} {a['name']} ({a['type']})  "
                f"エントリー:{a['entry_price']}円→現在:{a['exit_price']}円  リターン:{ret}"
            )
        body_parts.append("■ 決済アラート(損切り/利確/保有期間終了)\n" + "\n".join(lines))

    body_parts.append(
        "※ ルール: 日足でRSI25以下かつ出来高2倍以上。損切り-8%、利確+10%、保有は約3週間。\n"
        "※ 過去の結果は将来を保証しません。売買はご自身の判断で行ってください。"
    )
    body = "\n\n".join(body_parts)

    subject_parts = []
    if new_signals:
        subject_parts.append(f"買い{len(new_signals)}件")
    if exit_alerts:
        subject_parts.append(f"決済{len(exit_alerts)}件")
    subject = "【株シグナル通知】" + "・".join(subject_parts)

    send_email(subject, body)
    print(f"新規シグナル{len(new_signals)}件、決済アラート{len(exit_alerts)}件")


if __name__ == "__main__":
    main()
