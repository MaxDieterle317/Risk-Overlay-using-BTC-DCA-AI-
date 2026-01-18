# ----- Plotly interactive plots (replace matplotlib section) -----
# If you don't have plotly installed, run: pip install plotly
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load the saved backtest results
df = pd.read_csv("btc_dca_overlay_executed.csv", parse_dates=["Date"])

pio.renderers.default = "browser"  # try "notebook" if you're in Jupyter

# Prepare date x
x = df["Date"]

# 1) Net portfolio value interactive
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=x, y=df["net_value_baseline"], mode="lines", name="Baseline (net value)"))
fig1.add_trace(go.Scatter(x=x, y=df["net_value_overlay"], mode="lines", name="Overlay (net value)"))
fig1.update_layout(title="Baseline vs Overlay (Net Portfolio Value)",
                   xaxis_title="Date", yaxis_title="USD", template="plotly_white", height=600)
# Save and show
fig1.write_html("net_value_comparison.html", include_plotlyjs="cdn")
print("Saved interactive net value plot to net_value_comparison.html")
try:
    fig1.show()
except Exception:
    pass

# 2) Rolling Sharpe (30d) interactive
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x, y=rolling_sharpe_baseline, mode="lines", name="Baseline rolling Sharpe (30d)"))
fig2.add_trace(go.Scatter(x=x, y=rolling_sharpe_overlay, mode="lines", name="Overlay rolling Sharpe (30d)"))
fig2.update_layout(title="Rolling Sharpe (30-day window, annualized)",
                   xaxis_title="Date", yaxis_title="Annualized Sharpe", template="plotly_white", height=450)
fig2.write_html("rolling_sharpe_30d.html", include_plotlyjs="cdn")
print("Saved interactive rolling-sharpe plot to rolling_sharpe_30d.html")
try:
    fig2.show()
except Exception:
    pass

# 3) Short-window rolling Sharpe (7d) interactive
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=x, y=rs_base_7, mode="lines", name="Baseline rolling Sharpe (7d)"))
fig3.add_trace(go.Scatter(x=x, y=rs_over_7, mode="lines", name="Overlay rolling Sharpe (7d)"))
fig3.update_layout(title="Rolling Sharpe (7-day window, annualized)",
                   xaxis_title="Date", yaxis_title="Annualized Sharpe", template="plotly_white", height=450)
fig3.write_html("rolling_sharpe_7d.html", include_plotlyjs="cdn")
print("Saved interactive rolling-sharpe plot to rolling_sharpe_7d.html")
try:
    fig3.show()
except Exception:
    pass

# Optional: open files in codespace / local machine manually or click in file explorer
print("Plots saved. Open the .html files to interact with the charts.")
