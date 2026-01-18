# Risk-Overlay-using-BTC-DCA-AI-
This project explores how AI and machine learning can be combined with crypto market knowledge to create a risk-aware, long-term BTC accumulation strategy. The goal of this project is to learn how crypto markets work, how ML models are built and trained, and to gradually profit over time using given data.

# Key Principles

1. Small daily investments, no need to take risks when still learning.
2. Before building the model, define a clear, strict ruleset for it to follow.
3. Create simple models that are easier to explain to a general audience.
4. Realistic assumptions, dont expect to be a billionaire off this.

# Strategy 

Asset: Bitcoin (BTC)
Frequency: Daily
Amount: $1 per day
Behavior: Buy in accordance to MA, checking daily and adjusting accordingly ($1 maximum if volatility is high, pause if market crashes)

This baseline is used as a benchmark to evaluate whether added logic or ML actually improves outcomes.

# Trend Signal (20-Day Moving Average)

Instead of predicting prices directly, the system adjusts how aggressively it buys BTC based on market conditions.

 * Compute the 20-day moving average (MA20) of BTC price.
 * If price is above MA20 → market considered stronger.
 * If price is below MA20 → market considered weaker.

Trend Multiplier:
 
 * Above MA20 → 1.5× daily buy
 * Below MA20 → 0.5× daily buy

# Volatility Signal (20-Day vs 60-Day)

Volatility is used as a safety mechanism. Measure volatility over the last 20 days and meausre volatility over the last 60 days. If recent volatility is higher than long-term volatility → market is risky

Volatility Rule:

 * If volatility is high → cap daily buy at $1
 * If volatility is normal → no cap

# Final Rule-Based Policy

Daily Steps:

 1. Start with $1
 2. Apply trend multiplier (MA20)
 3. Apply volatility cap if needed
 4. Buy BTC

# ML Objective

One of the main goals of this project is to understand Machine Learning Principles to teach a model to learn the relationship between market conditions and optimal risk posture.

Inputs:
* Distance from MA20
* 20-day volatility
* 60-day volatility
* Volatility Ratio (20/60)
* Recent Returns

Targets:
* $0.00 (crash pause)
* Defensive (0.5x trend multiplier)
* Neutral (1x trend multiplier)
* Aggressive (1.5x trend multiplier)

Initial Models:
* Logistic Regression
* Decision Trees

# Roadmap

1. Gather BTC daily price data *
2. Implement baseline DCA *
3. Add rule-based risk overlay
4. Backtest against baseline
5. Generate ML training dataset
6. Train/eval ML models
7. Paper trade ML-driven strategy
8. Deploy with small real capital (minimum risk for now)

# Learning Objectives

* Use AI to help in cryptocurrency risk management.
* This project is primary for learning/research purposes.
* Crypto markets are notoriously voliatile, no profit guarantee
* Heavy testing before deploying real funds

# Techstack

* Python
* Pandas, numpy
* yfinance (yahoo finance for data acquisition)
* scikit-learn (for ML)
* matplotlib (graphing, data analysis/visualization)