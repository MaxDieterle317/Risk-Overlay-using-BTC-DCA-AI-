# 1) Install once (in your terminal / notebook cell)
# pip install yfinance pandas

import yfinance as yf
import pandas as pd

# 2) Download BTC-USD daily data
df = yf.download("BTC-USD", interval="1d", period="max", auto_adjust=True)

# 3) Keep only what we need (Close is enough for now)
if isinstance(df.columns, pd.MultiIndex):   # Flatten MultiIndex columns if present
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()[["Date", "Close", "Volume"]]

# 4) Basic sanity checks

print("\n********** BEGINNING PRICES **********")
print(df.head()) #beginning prices from BTC-USD

print("\n********** MOST RECENT PRICES **********")
print(df.tail()) #most recent prices from BTC-USD

print("\n********** MISSING VALUES IN EACH COLUMN *********")
print(df.isna().sum())

print("\n********** Dataframe Columns **********")
print(df.columns)

# --- Indicator calculations ---
df = df.sort_values("Date").reset_index(drop=True)

df["return"] = df["Close"].pct_change()

df["ma20"] = df["Close"].rolling(20).mean().shift(1)

df["vol20"] = df["return"].rolling(20).std().shift(1)
df["vol60"] = df["return"].rolling(60).std().shift(1)

df["high60"] = df["Close"].rolling(60).max().shift(1)
df["drawdown"] = (df["Close"] - df["high60"]) / df["high60"]

print(df[["Date", "Close", "ma20", "vol20", "vol60", "drawdown"]].head(70))

# --- Risk flags ---
df["above_ma20"] = df["Close"] > df["ma20"]
df["high_vol"] = df["vol20"] > df["vol60"]
df["crash"] = (~df["above_ma20"]) & df["high_vol"] & (df["drawdown"] <= -0.20)

# --- Buy amount logic ---
base = 1.0

# Start everyone at $1
df["daily_buy"] = base
df["usd_overlay"] = df["daily_buy"].cumsum()

# Trend multiplier (only when MA exists)
df.loc[df["ma20"].notna() & df["above_ma20"], "daily_buy"] = base * 1.5
df.loc[df["ma20"].notna() & (~df["above_ma20"]), "daily_buy"] = base * 0.5

# Volatility cap: if high volatility, maximum buy is $1
df.loc[df["high_vol"] & (df["daily_buy"] > 1.0), "daily_buy"] = 1.0

# Crash pause has highest priority: buy $0
df.loc[df["crash"], "daily_buy"] = 0.0

print("\n********** FULL RISK OVERLAY **********")
print(df["daily_buy"].value_counts().sort_index())

print ("\n********** CRASH DAY EXAMPLES **********") # One should see: You should see: daily_buy = 0.0, drawdown ≤ -0.20, vol20 > vol60, price below ma20
print(df.loc[df["crash"], ["Date", "Close", "ma20", "vol20", "vol60", "drawdown", "daily_buy"]].head(10))

# --- How much BTC did I get today for the dollars I spent? ---
df["btc_bought_overlay"] = df["daily_buy"] / df["Close"]
df["btc_bought_baseline"] = 1.0 / df["Close"]

# --- acc of BTC over time ---
df["btc_overlay"] = df["btc_bought_overlay"].cumsum()
df["btc_baseline"] = df["btc_bought_baseline"].cumsum()

# --- mult holdings by todays price ---
df["value_overlay"] = df["btc_overlay"] * df["Close"]
df["value_baseline"] = df["btc_baseline"] * df["Close"]

# --- capital investment ---
df["usd_baseline"] = 1.0
df["usd_baseline"] = df["usd_baseline"].cumsum()

print(df[[
    "Date",
    "daily_buy",
    "usd_overlay",
    "usd_baseline",
    "value_overlay",
    "value_baseline"
]].tail(5))
