"""
過去バックテスト: 「出来高急増+RSI」シグナルが過去に出ていたら、
2〜3週間保有した場合にどうなっていたかを検証するスクリプト。

使い方:
  python backtest.py

結果はメールで届きます（main.pyと同じGmail設定を使います）。
"""

import sys
import time
import pandas as pd
import yfinance as yf

from signal_logic import calc_rsi, calc_volume_ratio
from notifier import send_email

TICKERS_CSV = "tickers_nikkei225.csv"

# --- パラメータ ---
PERIOD = "4mo"          # 取得する日足データの期間（検証2ヶ月＋指標計算バッファ＋保有期間の余白）
LOOKBACK_DAYS = 40      # 過去何営業日分のシグナルを検証対象にするか（約2ヶ月）
HOLDING_DAYS = 15       # シグナル後、何営業日保有したと仮定するか（約3週間）
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
        interval="1d",
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


def find_past_signals(df: pd.DataFrame) -> list[dict]:
    """1銘柄分の日足dfから、過去のシグナルと保有後の結果を洗い出す"""
    df = df.dropna(how="all").copy()
    if "Close" not in df.columns or len(df) < RSI_PERIOD + VOLUME_WINDOW + HOLDING_DAYS + 2:
        return []

    df["rsi"] = calc_rsi(df["Close"], RSI_PERIOD)
    df["vol_ratio"] = calc_volume_ratio(df["Volume"], VOLUME_WINDOW)

    n = len(df)
    start_idx = max(RSI_PERIOD, VOLUME_WINDOW, n - LOOKBACK_DAYS - HOLDING_DAYS)
    end_idx = n - HOLDING_DAYS  # これ以降はholding_days分の未来データがない

    results = []
    for i in range(start_idx, end_idx):
        rsi_val = df["rsi"].iloc[i]
        vol_ratio = df["vol_ratio"].iloc[i]
        if pd.isna(rsi_val) or pd.isna(vol_ratio):
            continue
        if vol_ratio < VOLUME_SPIKE_TH:
            continue

        if rsi_val <= RSI_BUY_TH:
            signal_type = "BUY"
        elif rsi_val >= RSI_SELL_TH:
            signal_type = "SELL"
        else:
            continue

        entry_price = float(df["Close"].iloc[i])
        exit_price = float(df["Close"].iloc[i + HOLDING_DAYS])
        raw_return = (exit_price / entry_price) - 1.0
        ret = raw_return if signal_type == "BUY" else -raw_return

        results.append(
            {
                "date": df.index[i].strftime("%Y-%m-%d"),
                "type": signal_type,
                "rsi": round(float(rsi_val), 1),
                "volume_ratio": round(float(vol_ratio), 2),
                "entry_price": round(entry_price, 1),
                "exit_price": round(exit_price, 1),
                "return_pct": round(ret * 100, 2),
            }
        )
    return results


def main():
    tickers_df = load_tickers(TICKERS_CSV)
    symbols = tickers_df["yf_symbol"].tolist()

    all_results = []

    for i in range(0, len(symbols), BATCH_SIZE):
        batch_symbols = symbols[i : i + BATCH_SIZE]
        try:
            batch_data = fetch_batch(batch_symbols)
        except Exception as e:
            print(f"取得エラー(バッチ {i}): {e}", file=sys.stderr)
            continue

        for sym, df in batch_data.items():
            code = sym.replace(".T", "")
            name_row = tickers_df.loc[tickers_df["code"] == code, "name"]
            name = name_row.values[0] if len(name_row) else code

            signals = find_past_signals(df)
            for s in signals:
                s["code"] = code
                s["name"] = name
                all_results.append(s)

        time.sleep(1)

    if not all_results:
        send_email(
            "【バックテスト結果】過去2ヶ月、シグナルなし",
            "過去2ヶ月間、条件に合致するシグナルはありませんでした。",
        )
        print("シグナルなし")
        return

    result_df = pd.DataFrame(all_results).sort_values("date")

    total = len(result_df)
    win_count = (result_df["return_pct"] > 0).sum()
    win_rate = win_count / total * 100
    avg_return = result_df["return_pct"].mean()
    median_return = result_df["return_pct"].median()
    best = result_df.loc[result_df["return_pct"].idxmax()]
    worst = result_df.loc[result_df["return_pct"].idxmin()]

    summary_lines = [
        f"■ 検証条件: 過去{LOOKBACK_DAYS}営業日のシグナル / 保有{HOLDING_DAYS}営業日(約3週間)を想定",
        f"■ シグナル総数: {total}件",
        f"■ 勝率(プラスで終わった割合): {win_rate:.1f}%",
        f"■ 平均リターン: {avg_return:+.2f}%",
        f"■ 中央値リターン: {median_return:+.2f}%",
        f"■ 最も良かった例: {best['date']} {best['code']} {best['name']} "
        f"[{best['type']}] {best['return_pct']:+.2f}%",
        f"■ 最も悪かった例: {worst['date']} {worst['code']} {worst['name']} "
        f"[{worst['type']}] {worst['return_pct']:+.2f}%",
        "",
        "--- シグナル一覧(日付順) ---",
    ]

    detail_lines = []
    for _, row in result_df.iterrows():
        detail_lines.append(
            f"{row['date']} [{row['type']}] {row['code']} {row['name']}  "
            f"RSI:{row['rsi']}  出来高倍率:{row['volume_ratio']}倍  "
            f"リターン:{row['return_pct']:+.2f}%"
        )

    body = "\n".join(summary_lines + detail_lines)

    send_email(f"【バックテスト結果】シグナル{total}件・勝率{win_rate:.0f}%", body)
    print(f"バックテスト完了。シグナル{total}件、勝率{win_rate:.1f}%")


if __name__ == "__main__":
    main()
