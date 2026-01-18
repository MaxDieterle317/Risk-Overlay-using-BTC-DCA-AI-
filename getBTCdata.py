#!/usr/bin/env python3
"""
getBTCdata_executed.py
Single-file script: BTC DCA + Risk Overlay + Execution Simulator + Metrics + Plots

Dependencies:
    pip install yfinance pandas numpy matplotlib
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Config (tune these) ---
TICKER = "BTC-USD"
BASE_DAILY_USD = 1.0
MA_WINDOW = 20
VOL_SHORT = 20
VOL_LONG = 60
DRAWDOWN_WINDOW = 60
DRAWDOWN_THRESHOLD = 0.20   # 20% drawdown
FEE_PCT = 0.0015            # 0.15% per trade (notional)
SLIPPAGE_PCT = 0.001        # 0.1% slippage
MIN_TRADE_USD = 1.0
TICK_SIZE = 1e-8

# ---------------------------
# 1) Download data
# ---------------------------
df = yf.download(TICKER, interval="1d", period="max", auto_adjust=True)

# flatten MultiIndex columns if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# keep columns we need
df = df.reset_index()[["Date", "Close", "Volume"]]

# --- Basic checks
print("\n********** BEGINNING PRICES **********")
print(df.head())
print("\n********** MOST RECENT PRICES **********")
print(df.tail())
print("\n********** MISSING VALUES IN EACH COLUMN *********")
print(df.isna().sum())
print("\n********** Dataframe Columns **********")
print(df.columns)

# ---------------------------
# 2) Indicators (no lookahead)
# ---------------------------
df = df.sort_values("Date").reset_index(drop=True)
df["return"] = df["Close"].pct_change()

# MA20 shifted so today's decision uses yesterday's MA
df["ma20"] = df["Close"].rolling(MA_WINDOW).mean().shift(1)

# Volatility (std of daily returns), shifted
df["vol20"] = df["return"].rolling(VOL_SHORT).std().shift(1)
df["vol60"] = df["return"].rolling(VOL_LONG).std().shift(1)

# 60-day rolling high (shifted) and drawdown
df["high60"] = df["Close"].rolling(DRAWDOWN_WINDOW).max().shift(1)
df["drawdown"] = (df["Close"] - df["high60"]) / df["high60"]

print("\nSample indicators (first 70 rows):")
print(df[["Date", "Close", "ma20", "vol20", "vol60", "drawdown"]].head(70))

# ---------------------------
# 3) Risk flags & buy logic
# ---------------------------
df["above_ma20"] = df["Close"] > df["ma20"]
df["high_vol"] = df["vol20"] > df["vol60"]

# Crash condition (all three must be true)
df["crash"] = (~df["above_ma20"]) & df["high_vol"] & (df["drawdown"] <= -DRAWDOWN_THRESHOLD)

# Base daily buy - default (will be overwritten)
df["daily_buy"] = BASE_DAILY_USD

# Trend multiplier
df.loc[df["ma20"].notna() & df["above_ma20"], "daily_buy"] = BASE_DAILY_USD * 1.5
df.loc[df["ma20"].notna() & (~df["above_ma20"]), "daily_buy"] = BASE_DAILY_USD * 0.5

# Volatility cap: if high volatility, cap at base
df.loc[df["high_vol"] & (df["daily_buy"] > BASE_DAILY_USD), "daily_buy"] = BASE_DAILY_USD

# Crash pause has highest priority
df.loc[df["crash"], "daily_buy"] = 0.0

print("\n********** FULL RISK OVERLAY **********")
print(df["daily_buy"].value_counts().sort_index())

print("\n********** CRASH DAY EXAMPLES **********")
print(df.loc[df["crash"], ["Date", "Close", "ma20", "vol20", "vol60", "drawdown", "daily_buy"]].head(10))

# ---------------------------
# 4) Execution simulator (fees + slippage)
# ---------------------------
def execute_buy(usd_amount, price, fee_pct=FEE_PCT, slippage_pct=SLIPPAGE_PCT, min_trade_usd=MIN_TRADE_USD, tick_size=TICK_SIZE):
    if usd_amount is None or usd_amount <= 0 or usd_amount < min_trade_usd:
        return 0.0, 0.0
    exec_price = price * (1 + slippage_pct)
    fee_usd = usd_amount * fee_pct
    usd_after_fee = usd_amount - fee_usd
    btc = usd_after_fee / exec_price
    # round down to tick size
    btc = np.floor(btc / tick_size) * tick_size
    return float(btc), float(fee_usd)

# overlay execution
btc_bought_overlay_exec = []
fee_usd_overlay = []
for usd, price in zip(df["daily_buy"].fillna(0.0), df["Close"]):
    btc, fee = execute_buy(usd, price)
    btc_bought_overlay_exec.append(btc)
    fee_usd_overlay.append(fee)

df["btc_bought_overlay_exec"] = btc_bought_overlay_exec
df["fee_usd_overlay"] = fee_usd_overlay
df["btc_overlay_exec"] = df["btc_bought_overlay_exec"].cumsum()
df["fees_overlay_cum"] = df["fee_usd_overlay"].cumsum()
df["usd_overlay"] = df["daily_buy"].fillna(0.0).cumsum()

# baseline execution ($1 per day)
btc_bought_baseline_exec = []
fee_usd_baseline = []
for price in df["Close"]:
    btc, fee = execute_buy(1.0, price)
    btc_bought_baseline_exec.append(btc)
    fee_usd_baseline.append(fee)

df["btc_bought_baseline_exec"] = btc_bought_baseline_exec
df["fee_usd_baseline"] = fee_usd_baseline
df["btc_baseline_exec"] = df["btc_bought_baseline_exec"].cumsum()
df["fees_baseline_cum"] = pd.Series(df["fee_usd_baseline"]).cumsum()
df["usd_baseline"] = 1.0
df["usd_baseline"] = df["usd_baseline"].cumsum()

# mark-to-market and net values
df["value_overlay_exec"] = df["btc_overlay_exec"] * df["Close"]
df["value_baseline_exec"] = df["btc_baseline_exec"] * df["Close"]
df["net_value_overlay"] = df["value_overlay_exec"] - df["fees_overlay_cum"]
df["net_value_baseline"] = df["value_baseline_exec"] - df["fees_baseline_cum"]

# ---------------------------
# 5) Performance stats
# ---------------------------
def performance_stats(value_series, usd_invested_series):
    final_value = float(value_series.iloc[-1])
    total_invested = float(usd_invested_series.iloc[-1])
    roi = (final_value - total_invested) / total_invested if total_invested > 0 else np.nan

    daily_ret = value_series.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 1 else np.nan
    ann_ret = (1 + daily_ret.mean())**252 - 1 if len(daily_ret) > 0 else np.nan
    sharpe = ann_ret / ann_vol if (ann_vol and ann_vol > 0) else np.nan

    cummax = value_series.cummax()
    drawdown = (value_series - cummax) / cummax
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan

    return {
        "final_value": final_value,
        "total_invested": total_invested,
        "roi": roi,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd
    }

stats_overlay = performance_stats(df["net_value_overlay"], df["usd_overlay"])
stats_baseline = performance_stats(df["net_value_baseline"], df["usd_baseline"])

print("\n--- EXECUTION + PERFORMANCE SUMMARY ---")
print("Overlay stats:", stats_overlay)
print("Baseline stats:", stats_baseline)

summary = pd.DataFrame([stats_baseline, stats_overlay], index=["baseline","overlay"]).T
print("\nSummary table (baseline vs overlay):")
print(summary.round(4))

# ---------------------------
# 6) Final executed snapshot & checks
# ---------------------------
print("\n--- FINAL (EXECUTED) PORTFOLIO SNAPSHOT ---")
print(df[[
    "Date",
    "daily_buy",
    "usd_overlay",
    "usd_baseline",
    "net_value_overlay",
    "net_value_baseline"
]].tail(5))

# ---------------------------
# 7) Plots (comparison + rolling metrics)
# ---------------------------
plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["net_value_baseline"], label="Baseline (net value)", alpha=0.9)
plt.plot(df["Date"], df["net_value_overlay"], label="Overlay (net value)", alpha=0.9)
plt.legend()
plt.title("Baseline vs Overlay (Net Portfolio Value)")
plt.xlabel("Date")
plt.ylabel("USD")
plt.grid(True)
plt.tight_layout()
plt.show()

# Rolling Sharpe (30-day approx -> annualize)
rolling_window = 30
daily_ret_overlay = df["net_value_overlay"].pct_change().fillna(0)
rolling_sharpe_overlay = (daily_ret_overlay.rolling(rolling_window).mean() /
                          (daily_ret_overlay.rolling(rolling_window).std() + 1e-12)) * np.sqrt(252)

daily_ret_baseline = df["net_value_baseline"].pct_change().fillna(0)
rolling_sharpe_baseline = (daily_ret_baseline.rolling(rolling_window).mean() /
                           (daily_ret_baseline.rolling(rolling_window).std() + 1e-12)) * np.sqrt(252)
# --- Diagnostics + guaranteed plot saving/display (insert after computing rolling_sharpe_baseline & overlay) ---

# Inspect how many non-NaN values exist
print("\nRolling Sharpe diagnostics:")
print("baseline non-null count:", rolling_sharpe_baseline.dropna().shape[0])
print("overlay  non-null count:", rolling_sharpe_overlay.dropna().shape[0])
print("baseline sample (tail):")
print(rolling_sharpe_baseline.tail(10))
print("overlay sample (tail):")
print(rolling_sharpe_overlay.tail(10))

# If you want to see a shorter rolling window for quicker feedback, try 7 days:
rolling_window_short = 7
rs_base_7 = (daily_ret_baseline.rolling(rolling_window_short).mean() /
             (daily_ret_baseline.rolling(rolling_window_short).std() + 1e-12)) * np.sqrt(252)
rs_over_7 = (daily_ret_overlay.rolling(rolling_window_short).mean() /
             (daily_ret_overlay.rolling(rolling_window_short).std() + 1e-12)) * np.sqrt(252)
print("\nShort-window (7d) rolling Sharpe tail (baseline / overlay):")
print(rs_base_7.tail(6))
print(rs_over_7.tail(6))

# Recreate second plot but save to file so you can open it even if the viewer doesn't pop up
plt.figure(figsize=(12,4))
plt.plot(df["Date"], rolling_sharpe_baseline, label="Baseline rolling Sharpe (30d)", alpha=0.9)
plt.plot(df["Date"], rolling_sharpe_overlay, label="Overlay rolling Sharpe (30d)", alpha=0.9)
plt.legend()
plt.title("Rolling Sharpe (30-day window, annualized)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.savefig("rolling_sharpe_30d.png")
print("\nSaved rolling-sharpe plot to rolling_sharpe_30d.png")

# Also save the short-window version for quick visual confirmation
plt.figure(figsize=(12,4))
plt.plot(df["Date"], rs_base_7, label="Baseline rolling Sharpe (7d)", alpha=0.9)
plt.plot(df["Date"], rs_over_7, label="Overlay rolling Sharpe (7d)", alpha=0.9)
plt.legend()
plt.title("Rolling Sharpe (7-day window, annualized)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.savefig("rolling_sharpe_7d.png")
print("Saved short-window rolling-sharpe plot to rolling_sharpe_7d.png")

# Finally, show all figures (some backends only render the last; show() after saving forces rendering)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(df["Date"], rolling_sharpe_baseline, label="Baseline rolling Sharpe (30d)", alpha=0.9)
plt.plot(df["Date"], rolling_sharpe_overlay, label="Overlay rolling Sharpe (30d)", alpha=0.9)
plt.legend()
plt.title("Rolling Sharpe (30-day window, annualized)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# 8) Save a small CSV for later modeling/backtest analysis
# ---------------------------
out_cols = [
    "Date", "Close", "daily_buy", "btc_bought_overlay_exec", "btc_overlay_exec", "fees_overlay_cum",
    "btc_bought_baseline_exec", "btc_baseline_exec", "fees_baseline_cum",
    "net_value_overlay", "net_value_baseline", "ma20", "vol20", "vol60", "drawdown", "crash"
]
df[out_cols].to_csv("btc_dca_overlay_executed.csv", index=False)
print("\nSaved executed results to btc_dca_overlay_executed.csv")
