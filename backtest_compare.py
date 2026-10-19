"""
直近1か月シミュレーション: もし1か月前から新しい通知ルールで買っていたら、今いくらの損益か。

ルール(本番の main.py と同じ):
  日足で「RSI25以下」かつ「出来高が直近平均の2倍以上」になった銘柄を、翌朝の始値で買う。
  損切り-8%、利確+10%、最大15営業日保有。同じ銘柄は、前のポジションが終わるまで買わない。
  株式分割の影響を避けるため、分割調整済みの株価を使う。

前提:
  - 1件ごとに100万円ずつ買ったと考える(金額は INVEST_PER_TRADE で変更できる)
  - 手数料・税金・約定のズレは考慮しない
  - 損切り・利確は、その日の終値で判定する
  - 今まだ持っているもの(保有中)は、最新の終値で計算した含み損益

使い方:
  python backtest_compare.py             # 結果をメールで送る
  python backtest_compare.py --no-email  # 画面に表示するだけ
"""

import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import yfinance as yf

from signal_logic_v2 import OLD_CONFIG, calc_signals

TICKERS_CSV = "tickers_nikkei225.csv"

PERIOD = "8mo"            # 取得する期間(RSIなどの助走期間込み)
MONTH_DAYS = 30           # 何日前から始めたことにするか
HOLDING_DAYS = 15
STOP_LOSS_PCT = 0.08
TAKE_PROFIT_PCT = 0.10
INVEST_PER_TRADE = 1_000_000  # 1件あたりの投資額(円)
BATCH_SIZE = 50

# 買いのみ・RSI25以下・出来高2倍以上(本番と同じ)
CONFIG = replace(OLD_CONFIG, rsi_buy_th=25.0, volume_spike_th=2.0)


def load_tickers(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"code": str})
    df["yf_symbol"] = df["code"] + ".T"
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """欠損行を除き、日付を「時刻なし・タイムゾーンなし」にそろえる"""
    df = df.dropna(how="all").copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df


def fetch_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=" ".join(symbols),
        period=PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,  # 株式分割・配当を調整した株価
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


def simulate_recent(df: pd.DataFrame, signals: pd.Series, start_idx: int):
    """
    start_idx 以降のシグナルを、翌朝の始値で買った場合の結果にする。
    戻り値: (買った記録のリスト, 最新日にシグナルが出ている=明日の朝の買い候補か)
    """
    close = df["Close"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    sig = signals.to_numpy()
    n = len(df)

    trades = []
    next_ok = start_idx
    for i in range(start_idx, n - 1):  # 翌日(i+1)のデータが必要
        if i < next_ok or not sig[i]:
            continue
        entry = open_[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue

        exit_idx, reason = None, None
        for j in range(i + 1, min(i + HOLDING_DAYS, n - 1) + 1):
            r = close[j] / entry - 1.0
            if r <= -STOP_LOSS_PCT:
                exit_idx, reason = j, "損切り"
                break
            if r >= TAKE_PROFIT_PCT:
                exit_idx, reason = j, "利確"
                break
        if exit_idx is None:
            if i + HOLDING_DAYS <= n - 1:
                exit_idx, reason = i + HOLDING_DAYS, "期間満了"
            else:
                exit_idx, reason = n - 1, "保有中"

        trades.append(
            {
                "signal_date": df.index[i],
                "entry_date": df.index[i + 1],
                "entry_price": entry,
                "exit_date": df.index[exit_idx],
                "exit_price": close[exit_idx],
                "ret": close[exit_idx] / entry - 1.0,
                "reason": reason,
            }
        )
        next_ok = n if reason == "保有中" else exit_idx + 1

    candidate = bool(sig[n - 1]) and next_ok <= n - 1 and n - 1 >= start_idx
    return trades, candidate


def yen(x: float) -> str:
    """円をわかりやすい表記(万円)にする"""
    return f"{x / 10_000:+,.1f}万円"


def main():
    send_mail = "--no-email" not in sys.argv

    tickers_df = load_tickers(TICKERS_CSV)
    symbols = tickers_df["yf_symbol"].tolist()
    names = dict(zip(tickers_df["code"], tickers_df["name"]))

    all_data: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), BATCH_SIZE):
        try:
            batch = fetch_batch(symbols[i : i + BATCH_SIZE])
        except Exception as e:
            print(f"取得エラー(バッチ {i}): {e}", file=sys.stderr)
            continue
        for sym, df in batch.items():
            df = clean(df)
            if "Close" in df.columns and "Open" in df.columns and len(df) > 80:
                all_data[sym] = df
        time.sleep(1)
    print(f"取得できた銘柄数: {len(all_data)}")

    last_date = max(df.index[-1] for df in all_data.values())
    start_date = last_date - pd.Timedelta(days=MONTH_DAYS)

    trades, candidates = [], []
    for sym, df in all_data.items():
        code = sym.replace(".T", "")
        start_idx = int(df.index.searchsorted(start_date))
        signals = calc_signals(df, CONFIG)
        signals = signals.where(signals == "BUY", "")  # 買いだけを使う
        ts, cand = simulate_recent(df, signals, start_idx)
        for t in ts:
            t["code"], t["name"] = code, names.get(code, code)
            trades.append(t)
        if cand:
            candidates.append((code, names.get(code, code), float(df["Close"].iloc[-1])))

    trades.sort(key=lambda t: (t["signal_date"], t["code"]))
    closed = [t for t in trades if t["reason"] != "保有中"]
    holding = [t for t in trades if t["reason"] == "保有中"]

    pl_closed = sum(t["ret"] for t in closed) * INVEST_PER_TRADE
    pl_holding = sum(t["ret"] for t in holding) * INVEST_PER_TRADE
    wins = sum(1 for t in closed if t["ret"] > 0)

    # 同時に持っていた最大の件数(必要な資金の目安)
    days = pd.bdate_range(start_date, last_date)
    peak = 0
    for d in days:
        cnt = sum(1 for t in trades if t["entry_date"] <= d <= t["exit_date"])
        peak = max(peak, cnt)

    lines = [
        f"■ シミュレーション期間: {start_date.strftime('%Y-%m-%d')} 〜 {last_date.strftime('%Y-%m-%d')} "
        f"(1か月前から始めたと仮定)",
        "■ ルール: 日足でRSI25以下かつ出来高2倍以上 → 翌朝の始値で買う。損切り-8%、利確+10%、最大15営業日",
        f"■ 前提: 1件ごとに{INVEST_PER_TRADE // 10_000}万円ずつ買う。手数料・税金は含まない",
        "",
        f"■ 買った件数: {len(trades)}件(決済済み{len(closed)}件 / 保有中{len(holding)}件)",
        f"■ 合計損益: {yen(pl_closed + pl_holding)}",
        f"     うち決済済み: {yen(pl_closed)} / 保有中の含み損益: {yen(pl_holding)}",
    ]
    if closed:
        avg = np.mean([t["ret"] for t in closed]) * 100
        lines.append(f"■ 決済済みの勝率: {wins}/{len(closed)}件 = {wins / len(closed) * 100:.1f}%  平均リターン: {avg:+.2f}%")
    lines.append(f"■ 同時に持っていた最大の件数: {peak}件(必要な資金の目安: 約{peak * INVEST_PER_TRADE // 10_000}万円)")
    lines.append("")
    if candidates:
        lines.append("■ 明日の朝の買い候補(今日の大引けでシグナルが出た銘柄)")
        for code, name, price in candidates:
            lines.append(f"   {code} {name}  終値:{price:,.1f}円")
    else:
        lines.append("■ 明日の朝の買い候補: なし")
    lines += ["", "--- 1件ごとの結果(シグナルが出た日の順) ---"]
    for t in trades:
        lines.append(
            f"{t['signal_date'].strftime('%m/%d')}→{t['exit_date'].strftime('%m/%d')} "
            f"[{t['reason']}] {t['code']} {t['name']}  "
            f"買値:{t['entry_price']:,.1f}円 → {t['exit_price']:,.1f}円  "
            f"{t['ret'] * 100:+.2f}% ({yen(t['ret'] * INVEST_PER_TRADE)})"
        )
    lines += ["", "※ 過去の仮想の結果です。将来の成績を保証しません。売買はご自身の判断で行ってください。"]
    body = "\n".join(lines)
    print(body)

    if send_mail:
        from notifier import send_email

        send_email(
            f"【直近1か月シミュレーション】合計{yen(pl_closed + pl_holding)}(買い{len(trades)}件)", body
        )
        print("メールを送信しました")


if __name__ == "__main__":
    main()
