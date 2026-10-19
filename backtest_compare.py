"""
バックテスト比較 第3回: 「翌朝の始値で買った場合」の確認 (買いのみ・約4年)

第2回の結果、「RSI25以下・出来高2倍以上」が一番良かった。
ただし第2回は「シグナルが出た日の終値で買う」計算だった。実際の通知は大引け後に届くので、
買えるのは翌朝の始値になる。その場合でも成績が保たれるかを確かめる。

比べるパターン (すべて買いのみ・出来高の倍率つき):
  RSI30以下・出来高2倍 (従来ルール) / RSI25以下・出来高2倍 / RSI20以下・出来高2倍 / RSI25以下・出来高3倍
  それぞれを「当日終値で買う」「翌朝の始値で買う」の2通りで計算する。

共通の前提:
  - 最大15営業日保有。途中で -8% 以下になったら損切り、+10% 以上で利確 (終値で判定)
  - 同じ銘柄のシグナルは、前のポジションが終わるまで無視
  - 株式分割の影響を避けるため、分割調整済みの株価を使う
  - 手数料・税金・スリッページは考慮しない

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

from signal_logic_v2 import (
    OLD_CONFIG,
    SignalConfig,
    calc_signals,
)

TICKERS_CSV = "tickers_nikkei225.csv"

PERIOD = "5y"          # 取得する期間(指標の助走期間込み)
LOOKBACK_DAYS = 1000   # 検証する期間(営業日。約4年)
HOLDING_DAYS = 15
STOP_LOSS_PCT = 0.08
TAKE_PROFIT_PCT = 0.10
BATCH_SIZE = 50
MIN_RELIABLE_TRADES = 30  # これより少ないと偶然の可能性が高い


CONDITIONS = [(30.0, 2.0), (25.0, 2.0), (20.0, 2.0), (25.0, 3.0)]


def condition_name(rsi: float, vol: float) -> str:
    name = f"RSI{rsi:.0f}以下・出来高{vol:.0f}倍以上"
    tags = {(30, 2): "従来ルール", (25, 2): "有力候補", (25, 3): "案4"}
    tag = tags.get((round(rsi), round(vol)))
    return f"{name} [{tag}]" if tag else name


def build_config(rsi: float, vol: float) -> SignalConfig:
    return replace(OLD_CONFIG, rsi_buy_th=rsi, volume_spike_th=vol)


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


def simulate(df: pd.DataFrame, signals: pd.Series, entry_mode: str = "close") -> list[dict]:
    """
    シグナルの出た日ごとに、保有した場合の結果を計算する。
    entry_mode="close"     : シグナルが出た日の終値で買う
    entry_mode="next_open" : 翌営業日の始値で買う (大引け後に通知が届く実際の流れ)
    どちらも、シグナルの翌営業日から15営業日ぶんの終値を見て、損切り・利確・期間満了を判定する。
    """
    close = df["Close"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    sig = signals.to_numpy()
    n = len(df)

    first = max(0, n - LOOKBACK_DAYS - HOLDING_DAYS)
    last = n - HOLDING_DAYS - 1  # 保有期間ぶんのデータが残っている日まで
    next_ok = first
    trades = []

    for i in range(first, last + 1):
        if i < next_ok or not sig[i]:
            continue

        entry = close[i] if entry_mode == "close" else open_[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue

        side = sig[i]
        exit_idx = i + HOLDING_DAYS
        reason = "期間満了"

        for j in range(i + 1, i + HOLDING_DAYS + 1):
            r = close[j] / entry - 1.0
            r = r if side == "BUY" else -r
            if r <= -STOP_LOSS_PCT:
                exit_idx, reason = j, "損切り"
                break
            if r >= TAKE_PROFIT_PCT:
                exit_idx, reason = j, "利確"
                break

        r = close[exit_idx] / entry - 1.0
        r = r if side == "BUY" else -r
        trades.append(
            {
                "date": df.index[i],
                "type": side,
                "return_pct": r * 100.0,
                "reason": reason,
            }
        )
        next_ok = exit_idx + 1

    return trades


def summarize(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "0件"
    n = len(trades)
    win = (trades["return_pct"] > 0).mean() * 100
    mean = trades["return_pct"].mean()
    median = trades["return_pct"].median()
    worst = trades["return_pct"].min()
    stops = (trades["reason"] == "損切り").sum()
    note = "  ※件数が少なく参考程度" if n < MIN_RELIABLE_TRADES else ""
    return (
        f"{n}件  勝率{win:.1f}%  平均{mean:+.2f}%  中央値{median:+.2f}%  "
        f"最悪{worst:+.1f}%  損切り{stops}件{note}"
    )


def main():
    send_mail = "--no-email" not in sys.argv

    tickers_df = load_tickers(TICKERS_CSV)
    symbols = tickers_df["yf_symbol"].tolist()

    all_data: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), BATCH_SIZE):
        try:
            batch = fetch_batch(symbols[i : i + BATCH_SIZE])
        except Exception as e:
            print(f"取得エラー(バッチ {i}): {e}", file=sys.stderr)
            continue
        for sym, df in batch.items():
            df = clean(df)
            if "Close" in df.columns and len(df) > 80:
                all_data[sym] = df
        time.sleep(1)

    print(f"取得できた銘柄数: {len(all_data)}")

    # 前半・後半に分ける日付(検証期間のちょうど真ん中)
    longest = max(all_data.values(), key=len)
    mid_pos = max(0, len(longest) - LOOKBACK_DAYS // 2 - HOLDING_DAYS)
    cutoff = longest.index[mid_pos]

    MODES = [("close", "当日終値で買う(第2回と同じ)"), ("next_open", "翌朝の始値で買う(実際の流れ)")]
    results: dict[tuple, pd.DataFrame] = {}
    for rsi, vol in CONDITIONS:
        cfg = build_config(rsi, vol)
        per_mode = {m: [] for m, _ in MODES}
        for sym, df in all_data.items():
            signals = calc_signals(df, cfg)
            signals = signals.where(signals == "BUY", "")  # 買いだけを使う
            for m, _ in MODES:
                for t in simulate(df, signals, m):
                    t["code"] = sym.replace(".T", "")
                    per_mode[m].append(t)
        for m, _ in MODES:
            results[(rsi, vol, m)] = pd.DataFrame(
                per_mode[m], columns=["date", "type", "return_pct", "reason", "code"]
            )

    lines = [
        f"■ 検証期間: 直近約{LOOKBACK_DAYS}営業日(約4年) / 対象{len(all_data)}銘柄 / 「買い」のみ",
        f"■ 前半・後半の境目: {cutoff.strftime('%Y-%m-%d')}",
        f"■ 保有: 最大{HOLDING_DAYS}営業日、損切り-{STOP_LOSS_PCT*100:.0f}%、利確+{TAKE_PROFIT_PCT*100:.0f}%",
        "■ 見方: 「翌朝の始値で買う」でも、平均リターンがプラスのまま保たれていれば安心できる。",
        "        件数が30件未満のものは偶然の可能性が高い。",
        "",
    ]
    for rsi, vol in CONDITIONS:
        lines.append(f"【{condition_name(rsi, vol)}】")
        means = {}
        for m, mlabel in MODES:
            trades = results[(rsi, vol, m)]
            before = trades[trades["date"] < cutoff]
            after = trades[trades["date"] >= cutoff]
            means[m] = trades["return_pct"].mean() if len(trades) else float("nan")
            lines.append(f"  {mlabel}")
            lines.append(f"    全期間: {summarize(trades)}")
            lines.append(f"    前半  : {summarize(before)}")
            lines.append(f"    後半  : {summarize(after)}")
        diff = means["next_open"] - means["close"]
        lines.append(f"  → 平均リターンの差(始値買い - 終値買い): {diff:+.2f}%")
        lines.append("")

    lines += [
        "※ 手数料・税金・約定のズレは含まれていません。過去の結果は将来を保証しません。",
    ]
    body = "\n".join(lines)
    print(body)

    if send_mail:
        from notifier import send_email

        send_email("【バックテスト比較3】翌朝の始値で買った場合(買いのみ・約4年)", body)
        print("メールを送信しました")


if __name__ == "__main__":
    main()
