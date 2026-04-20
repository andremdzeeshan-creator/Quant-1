# Cross-Asset Time-Series Momentum Strategy

> Institutional-grade systematic trading strategy · Runs entirely in the browser · No server required

Based on **Moskowitz, Ooi & Pedersen (2012)** — *"Time Series Momentum"*, Journal of Financial Economics.

---

## What This Is

A fully systematic, cross-asset momentum strategy backtested across 7 instruments (US equities, bonds, commodities) from 2016–2026 March. Built to institutional standards with:

- Volatility-scaled position sizing (equal risk contribution per asset)
- Proper look-ahead bias prevention (1-period execution lag)
- Full performance attribution (Sharpe, Sortino, Calmar, VaR, drawdown)
- Monte Carlo robustness testing (500-path block bootstrap)
- Pairs trading overlay (SPY/QQQ cointegration signal)
- Interactive 6-phase research report

## Live Demo

Open `index.html` in any modern browser. The entire Python backtest runs client-side via [Pyodide](https://pyodide.org) (Python in WebAssembly). No install, no server, no API keys.

## Repo Structure

```
tsmom-strategy/
├── index.html          ← Full interactive dashboard (open this)
├── strategy.py         ← Pure Python strategy engine
├── requirements.txt    ← Python dependencies (for local use)
└── README.md
```

## Running Locally (Python)

```bash
pip install -r requirements.txt
python strategy.py
```

This exports `backtest_results.json` with all computed metrics.

## Strategy Summary

| Parameter | Value |
|---|---|
| Universe | SPY, QQQ, EEM, TLT, HYG, GLD, USO |
| Rebalance | Monthly (end-of-month) |
| Signal | 12-1 month lookback return |
| Sizing | Volatility-scaled (σ_target = 15%) |
| Transaction Cost | 10bps one-way |
| Period | Jan 2016 – March 2026 |

## Signal Mathematics

```
r(L)_it  = P_{i,t-1} / P_{i,t-12} − 1          # 12-1 month lookback return
s_it     = sign(r(L)_it)                          # direction signal ∈ {-1, +1}
σ_it     = std(r_i, t-60:t) × √12                # annualised realised vol
w_it     = s_it × σ_target / σ_it                # volatility-scaled weight
w̃_it     = w_it / Σⱼ|w_jt|                       # normalised position
R_t      = Σᵢ w̃_{i,t-1} × r_it − TC × Σᵢ|Δw̃_it| # net portfolio return
```

## Academic Reference

Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012).
*Time series momentum.* Journal of Financial Economics, 104(2), 228-250.

---

*Synthetic data calibrated to historical parameters. Not financial advice. For educational and interview purposes only.*
