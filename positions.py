"""
ポジション(保有想定)管理モジュール。

main.pyから呼び出される。シグナルが出た銘柄を「保有中」として
positions.csv というファイルに記録し、
- 保有中の銘柄には新しいシグナルを出さない（重複無視）
- 価格が-8%悪化したら「損切り」として決済する
- 価格が+10%上昇したら「利確」として決済する
- 保有からおよそ3週間(21日)経ったら「保有期間終了」として決済する
という管理を行う。
"""

import os
from datetime import datetime

import pandas as pd

POSITIONS_CSV = "positions.csv"
HOLDING_CALENDAR_DAYS = 21  # 保有期間の目安（約3週間）
STOP_LOSS_PCT = 0.08        # 損切りライン（8%）
TAKE_PROFIT_PCT = 0.10      # 利確ライン（10%）

COLUMNS = [
    "code",
    "name",
    "type",
    "entry_date",
    "entry_price",
    "status",
    "exit_date",
    "exit_price",
    "return_pct",
]


def load_positions() -> pd.DataFrame:
    if os.path.exists(POSITIONS_CSV):
        df = pd.read_csv(POSITIONS_CSV, dtype=str).fillna("")
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def save_positions(df: pd.DataFrame) -> None:
    df.to_csv(POSITIONS_CSV, index=False)


def has_open_position(positions_df: pd.DataFrame, code: str) -> bool:
    if positions_df.empty:
        return False
    open_rows = positions_df[
        (positions_df["code"] == code) & (positions_df["status"] == "open")
    ]
    return len(open_rows) > 0


def open_position(
    positions_df: pd.DataFrame, code: str, name: str, signal_type: str, price: float
) -> pd.DataFrame:
    new_row = {
        "code": code,
        "name": name,
        "type": signal_type,
        "entry_date": datetime.now().strftime("%Y-%m-%d"),
        "entry_price": round(price, 1),
        "status": "open",
        "exit_date": "",
        "exit_price": "",
        "return_pct": "",
    }
    return pd.concat([positions_df, pd.DataFrame([new_row])], ignore_index=True)


def check_exits(
    positions_df: pd.DataFrame, latest_prices: dict
) -> tuple[pd.DataFrame, list[dict]]:
    """保有中のポジションをチェックし、損切り・利確・保有期間終了に該当するものを決済する"""
    alerts = []
    today = datetime.now().date()

    for idx, row in positions_df.iterrows():
        if row["status"] != "open":
            continue
        code = row["code"]
        if code not in latest_prices:
            continue

        entry_price = float(row["entry_price"])
        latest_price = latest_prices[code]
        signal_type = row["type"]

        raw_return = (latest_price / entry_price) - 1.0
        signed_return = raw_return if signal_type == "BUY" else -raw_return

        entry_date = datetime.strptime(row["entry_date"], "%Y-%m-%d").date()
        days_held = (today - entry_date).days

        exit_reason = None
        if signed_return <= -STOP_LOSS_PCT:
            exit_reason = "損切り"
        elif signed_return >= TAKE_PROFIT_PCT:
            exit_reason = "利確"
        elif days_held >= HOLDING_CALENDAR_DAYS:
            exit_reason = "保有期間終了"

        if exit_reason:
            positions_df.at[idx, "status"] = "closed"
            positions_df.at[idx, "exit_date"] = today.strftime("%Y-%m-%d")
            positions_df.at[idx, "exit_price"] = str(round(latest_price, 1))
            positions_df.at[idx, "return_pct"] = str(round(signed_return * 100, 2))
            alerts.append(
                {
                    "code": code,
                    "name": row["name"],
                    "type": signal_type,
                    "reason": exit_reason,
                    "entry_price": entry_price,
                    "exit_price": round(latest_price, 1),
                    "return_pct": round(signed_return * 100, 2),
                }
            )

    return positions_df, alerts
