"""
売買シグナル判定ロジック
- 出来高急増（直近平均比）
- RSI（モメンタム・売られすぎ/買われすぎ）
の組み合わせでシグナルを判定する。
"""

import pandas as pd


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI(相対力指数)を計算する"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """直近平均出来高に対する倍率を計算する（当該バーは平均計算から除外）"""
    avg_volume = volume.shift(1).rolling(window=window, min_periods=window).mean()
    ratio = volume / avg_volume.replace(0, 1e-10)
    return ratio


def judge_signal(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_buy_th: float = 30.0,
    rsi_sell_th: float = 70.0,
    volume_window: int = 20,
    volume_spike_th: float = 2.0,
) -> dict | None:
    """
    df: 'Close', 'Volume' 列を持つ時系列データ（古い順）
    戻り値: シグナルがあれば dict、なければ None
    """
    if len(df) < max(rsi_period, volume_window) + 2:
        return None

    df = df.copy()
    df["rsi"] = calc_rsi(df["Close"], rsi_period)
    df["vol_ratio"] = calc_volume_ratio(df["Volume"], volume_window)

    last = df.iloc[-1]
    rsi_val = last["rsi"]
    vol_ratio = last["vol_ratio"]

    if pd.isna(rsi_val) or pd.isna(vol_ratio):
        return None

    is_volume_spike = vol_ratio >= volume_spike_th

    if not is_volume_spike:
        return None

    if rsi_val <= rsi_buy_th:
        return {
            "type": "BUY",
            "rsi": round(float(rsi_val), 1),
            "volume_ratio": round(float(vol_ratio), 2),
            "price": round(float(last["Close"]), 1),
        }
    elif rsi_val >= rsi_sell_th:
        return {
            "type": "SELL",
            "rsi": round(float(rsi_val), 1),
            "volume_ratio": round(float(vol_ratio), 2),
            "price": round(float(last["Close"]), 1),
        }

    return None
