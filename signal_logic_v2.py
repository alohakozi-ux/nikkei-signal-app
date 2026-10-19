"""
売買シグナル判定ロジック v2 (日足ベース)

従来のロジック(出来高急増 + RSI)に、的中率を上げるための5つの工夫を足したもの。
  案1 トレンド除外   : 25日移動平均から離れすぎた銘柄は見送る
  案2 反発確認       : RSIが極端な値になったあと、通常の範囲に戻ってきたら通知する
  案3 市場急落日除外 : 日経平均が大きく下げた日は見送る
  案4 しきい値強化   : RSI 25/75、出来高3倍にする
  案5 決算前後除外   : 決算発表の時期と、決算直後らしい値動きの銘柄は見送る

backtest(過去検証)と本番(main)の両方でこの同じ関数を使うので、
「検証した条件」と「実際に通知される条件」が必ず一致する。
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    # --- 基本条件 (案4: しきい値強化はここの数字) ---
    rsi_period: int = 14
    rsi_buy_th: float = 25.0     # 従来 30
    rsi_sell_th: float = 75.0    # 従来 70
    volume_window: int = 20
    volume_spike_th: float = 3.0  # 従来 2.0

    # --- 案1: トレンド除外 ---
    use_trend_filter: bool = True
    ma_window: int = 25
    ma_deviation_limit: float = 0.12  # 25日線から12%以上離れていたら見送る

    # --- 案2: 反発確認 ---
    use_rebound_confirm: bool = True
    rebound_lookback: int = 10  # 直近10営業日以内に極端なRSI+出来高急増があったか

    # --- 案3: 市場急落日除外 ---
    use_market_filter: bool = True
    market_drop_th: float = -0.02  # 日経平均が前日比 -2% 以下の日は見送る

    # --- 案5: 決算前後除外 ---
    use_earnings_filter: bool = True
    earnings_gap_th: float = 0.05      # 前日終値から5%以上のギャップ
    earnings_gap_volume: float = 3.0   # かつ出来高3倍以上 → 決算などの材料が出た日とみなす
    earnings_recent_days: int = 5      # その日から5営業日は見送る


# 従来ルール(比較用の基準)。工夫はすべてオフ。
OLD_CONFIG = SignalConfig(
    rsi_buy_th=30.0,
    rsi_sell_th=70.0,
    volume_spike_th=2.0,
    use_trend_filter=False,
    use_rebound_confirm=False,
    use_market_filter=False,
    use_earnings_filter=False,
)


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI(相対力指数)。従来ロジックと同じ計算方法。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calc_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """直近平均出来高に対する倍率(当日は平均から除外)。従来ロジックと同じ。"""
    avg_volume = volume.shift(1).rolling(window=window, min_periods=window).mean()
    return volume / avg_volume.replace(0, 1e-10)


def calc_market_drop(index_df: pd.DataFrame, threshold: float) -> pd.Series:
    """日経平均が threshold 以下の下落だった日を True にした Series(日付がindex)。"""
    ret = index_df["Close"].pct_change()
    return ret <= threshold


def in_earnings_season(index: pd.DatetimeIndex) -> pd.Series:
    """
    決算発表が集中する時期かどうか。
    3月決算の会社が多いため、おおよそ 1/25-2/14, 4/25-5/15, 7/25-8/14, 10/25-11/14。
    """
    md = pd.Series(index.month * 100 + index.day, index=index)
    return (
        ((md >= 125) & (md <= 214))
        | ((md >= 425) & (md <= 515))
        | ((md >= 725) & (md <= 814))
        | ((md >= 1025) & (md <= 1114))
    )


def calc_signals(
    df: pd.DataFrame,
    cfg: SignalConfig = SignalConfig(),
    market_drop: pd.Series | None = None,
) -> pd.Series:
    """
    日足データ全体について、各日が 'BUY' / 'SELL' / '' のどれかを返す。
    df: 'Open','Close','Volume' 列を持つ日足(古い順、indexは日付)
    market_drop: calc_market_drop() の結果(案3用)。省略すると案3は効かない。
    """
    close = df["Close"]
    rsi = calc_rsi(close, cfg.rsi_period)
    vol_ratio = calc_volume_ratio(df["Volume"], cfg.volume_window)
    spike = vol_ratio >= cfg.volume_spike_th

    buy_zone = rsi <= cfg.rsi_buy_th
    sell_zone = rsi >= cfg.rsi_sell_th

    if cfg.use_rebound_confirm:
        # 案2: 「極端なRSI + 出来高急増」の日が直近にあり、
        #      今日RSIが通常の範囲に戻ってきた(昨日は極端、今日は戻った)日に通知
        n = cfg.rebound_lookback
        buy_trigger = (buy_zone & spike).astype(float)
        sell_trigger = (sell_zone & spike).astype(float)
        buy_recent = buy_trigger.shift(1).rolling(n, min_periods=1).max() > 0
        sell_recent = sell_trigger.shift(1).rolling(n, min_periods=1).max() > 0
        buy = buy_recent & (rsi > cfg.rsi_buy_th) & (rsi.shift(1) <= cfg.rsi_buy_th)
        sell = sell_recent & (rsi < cfg.rsi_sell_th) & (rsi.shift(1) >= cfg.rsi_sell_th)
    else:
        buy = buy_zone & spike
        sell = sell_zone & spike

    if cfg.use_trend_filter:
        # 案1: 25日線から離れすぎ(急落中・急騰中)の銘柄は見送る
        ma = close.rolling(cfg.ma_window, min_periods=cfg.ma_window).mean()
        deviation = close / ma - 1.0
        buy = buy & (deviation >= -cfg.ma_deviation_limit)
        sell = sell & (deviation <= cfg.ma_deviation_limit)

    if cfg.use_market_filter and market_drop is not None:
        # 案3: 日経平均が急落した日は見送る
        ok = ~market_drop.reindex(df.index).fillna(False).astype(bool)
        buy = buy & ok
        sell = sell & ok

    if cfg.use_earnings_filter:
        # 案5: 決算シーズン中、または決算直後らしい値動き(大きなギャップ+出来高急増)から
        #      数営業日以内の銘柄は見送る
        season = in_earnings_season(df.index)
        gap = (df["Open"] / close.shift(1) - 1.0).abs()
        event = ((gap >= cfg.earnings_gap_th) & (vol_ratio >= cfg.earnings_gap_volume)).astype(float)
        recent_event = event.rolling(cfg.earnings_recent_days, min_periods=1).max() > 0
        ok = ~(season | recent_event)
        buy = buy & ok
        sell = sell & ok

    result = pd.Series("", index=df.index, dtype=object)
    result[buy.fillna(False).astype(bool)] = "BUY"
    result[sell.fillna(False).astype(bool)] = "SELL"
    return result


def judge_signal(
    df: pd.DataFrame,
    cfg: SignalConfig = SignalConfig(),
    market_drop: pd.Series | None = None,
) -> dict | None:
    """最新の1日分についてシグナルを判定する(本番用)。なければ None。"""
    min_len = max(cfg.rsi_period, cfg.volume_window, cfg.ma_window) + cfg.rebound_lookback + 2
    if len(df) < min_len:
        return None

    signals = calc_signals(df, cfg, market_drop)
    last_signal = signals.iloc[-1]
    if not last_signal:
        return None

    rsi_val = calc_rsi(df["Close"], cfg.rsi_period).iloc[-1]
    vol_ratio = calc_volume_ratio(df["Volume"], cfg.volume_window).iloc[-1]
    return {
        "type": last_signal,
        "rsi": round(float(rsi_val), 1),
        "volume_ratio": round(float(vol_ratio), 2),
        "price": round(float(df["Close"].iloc[-1]), 1),
    }
