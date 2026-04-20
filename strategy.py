"""
CROSS-ASSET TIME-SERIES MOMENTUM STRATEGY
==========================================
Institutional-grade systematic strategy implementation.
Based on Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"

Author: Quant Research Framework
Assets: US Equities ETFs, Crypto proxies, Commodities, Bonds
Timeframe: 2016–2026 March (monthly rebalancing)
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# SECTION 1: SYNTHETIC DATA GENERATION
# We use statistically realistic synthetic data
# calibrated to actual asset class properties.
# In production: replace with yfinance/Bloomberg.
# ─────────────────────────────────────────────

np.random.seed(42)  # Reproducibility — critical for backtesting

def generate_asset_prices(
    name, annual_return, annual_vol, start_price,
    n_months, trend_strength=0.3, mean_revert=0.05
):
    """
    Generate synthetic monthly price series with realistic properties:
    - Momentum: autocorrelation via trend_strength parameter
    - Fat tails: Student-t distributed shocks (df=5)
    - Volatility clustering: simplified GARCH-like effect
    
    Parameters
    ----------
    annual_return : float  — expected annual return (e.g. 0.10 = 10%)
    annual_vol    : float  — annualised volatility (e.g. 0.20 = 20%)
    trend_strength: float  — autocorrelation coefficient [0, 1]
    mean_revert   : float  — mean reversion speed [0, 1]
    """
    monthly_return = annual_return / 12
    monthly_vol = annual_vol / np.sqrt(12)
    
    returns = []
    prev_return = 0
    
    for _ in range(n_months):
        # Student-t shock: fatter tails than normal distribution
        # df=5 is standard for financial returns
        shock = np.random.standard_t(df=5) * monthly_vol
        
        # Trend component: partial autocorrelation
        trend_component = trend_strength * prev_return
        
        # Mean reversion component: pulls back to long-run mean
        reversion_component = mean_revert * (monthly_return - prev_return)
        
        r = monthly_return + trend_component + reversion_component + shock
        returns.append(r)
        prev_return = r
    
    # Convert returns to price series
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    
    return prices[1:]  # Drop initial price, keep n_months prices


def build_dataset():
    """
    Build cross-asset dataset spanning 2015–2024 (120 months).
    
    Asset universe (7 instruments across 4 asset classes):
    ─ Equities:    SPY (US Large Cap), QQQ (Tech), EEM (EM)
    ─ Bonds:       TLT (Long Treasury), HYG (High Yield)  
    ─ Commodities: GLD (Gold), USO (Oil)
    
    Each asset has calibrated return/vol params from historical data.
    """
    n_months = 123  # Jan 2016 – March 2026 = 123 months
    
    # Date index: month-end dates
    dates = pd.date_range(start='2016-01-31', periods=n_months, freq='ME')
    
    # Asset specifications: (annual_return, annual_vol, start_price, trend_strength)
    # Calibrated to approximate real historical properties
    asset_specs = {
        'SPY':  (0.12, 0.16, 200, 0.25),   # US Large Cap: high momentum
        'QQQ':  (0.18, 0.22, 100, 0.30),   # Tech: strong trend, higher vol
        'EEM':  (0.04, 0.20, 40,  0.15),   # EM: weak return, high vol, low momentum
        'TLT':  (0.02, 0.12, 120, 0.10),   # Long Bonds: low return, negative corr
        'HYG':  (0.05, 0.08, 85,  0.12),   # High Yield: credit spread dynamics
        'GLD':  (0.06, 0.13, 110, 0.20),   # Gold: moderate trend, safe-haven
        'USO':  (0.00, 0.35, 20,  0.18),   # Oil: near-zero return, very high vol
    }
    
    price_data = {}
    for ticker, (ret, vol, price, trend) in asset_specs.items():
        price_data[ticker] = generate_asset_prices(
            ticker, ret, vol, price, n_months, trend
        )
    
    prices = pd.DataFrame(price_data, index=dates)
    
    # Compute monthly returns: r_t = P_t/P_{t-1} - 1
    returns = prices.pct_change().dropna()
    
    return prices, returns


# ─────────────────────────────────────────────
# SECTION 2: SIGNAL GENERATION
# ─────────────────────────────────────────────

def compute_lookback_returns(prices, lookback=12, skip=1):
    """
    Compute (L - skip) month lookback return for each asset.
    
    The 'skip' parameter is critical: we skip the most recent month
    because 1-month returns exhibit REVERSAL (not momentum).
    Including it would contaminate our momentum signal.
    
    Formula: r^(L)_{i,t} = P_{i,t-skip} / P_{i,t-L} - 1
    
    Parameters
    ----------
    lookback : int — lookback window in months (default 12)
    skip     : int — months to skip at end (default 1)
    """
    # Shift prices by 'skip' months, then compute lookback return
    # prices.shift(skip) gives us P_{t-skip}
    # prices.shift(lookback) gives us P_{t-lookback}
    
    price_skip = prices.shift(skip)        # P_{t-1}  (numerator)
    price_back = prices.shift(lookback)    # P_{t-12} (denominator)
    
    lookback_returns = price_skip / price_back - 1
    return lookback_returns


def compute_realised_volatility(returns, window=60):
    """
    Compute rolling realised volatility (annualised).
    
    σ_{i,t} = std(r_{i,t-window:t}) × √12
    
    This is used for VOLATILITY SCALING — the key institutional
    technique that ensures each asset contributes equal risk.
    
    Why not use implied vol? Because:
    1. Not available for all assets
    2. Realised vol is sufficient for monthly rebalancing
    3. Implied vol introduces forward-looking bias in backtest
    
    Parameters
    ----------
    window : int — rolling window in months (default 60 = 5 years)
    """
    # .rolling(window).std() computes rolling standard deviation
    # Multiply by sqrt(12) to annualise from monthly to annual
    return returns.rolling(window=window).std() * np.sqrt(12)


def generate_tsmom_signals(prices, returns, lookback=12, skip=1, vol_window=60):
    """
    Generate Time-Series Momentum signals with volatility scaling.
    
    Full signal construction:
    
    Step 1: Compute lookback return → direction signal
        s_{i,t} = sign(r^(L)_{i,t})  ∈ {-1, +1}
    
    Step 2: Compute realised volatility
        σ_{i,t} = rolling std of returns (annualised)
    
    Step 3: Volatility-scale the position
        w_{i,t} = s_{i,t} × σ_target / σ_{i,t}
    
    Step 4: Clip to [-1.5, 1.5] to prevent extreme leverage
    
    Returns
    -------
    signals  : DataFrame — raw directional signals (-1 or +1)
    positions: DataFrame — volatility-scaled positions
    """
    sigma_target = 0.15  # 15% annualised vol target per asset
    
    # Step 1: Lookback returns → direction
    lookback_rets = compute_lookback_returns(prices, lookback, skip)
    
    # np.sign() returns -1, 0, or +1
    # We map 0 → 1 (no position defaults to flat, treated as long here)
    signals = np.sign(lookback_rets)
    signals = signals.replace(0, 1)  # Treat zero as flat/long
    
    # Step 2: Realised volatility
    realised_vol = compute_realised_volatility(returns, vol_window)
    
    # Step 3: Volatility scaling
    # Divide σ_target by realised_vol to get the scaling factor
    # Higher vol asset → smaller position (less weight)
    vol_scale = sigma_target / realised_vol
    
    # Step 4: Compute scaled positions
    positions = signals * vol_scale
    
    # Clip extreme positions: prevents runaway leverage
    positions = positions.clip(-1.5, 1.5)
    
    # Normalise each row so total gross exposure = 1
    # This is portfolio-level position sizing
    row_sum = positions.abs().sum(axis=1)
    positions_normalised = positions.div(row_sum, axis=0)
    
    return signals, positions_normalised, realised_vol


# ─────────────────────────────────────────────
# SECTION 3: TRADE SIMULATION
# ─────────────────────────────────────────────

def simulate_trades(returns, positions, transaction_cost=0.001):
    """
    Simulate portfolio returns from position signals.
    
    Core formula:
        Portfolio Return_t = Σ_i w_{i,t-1} × r_{i,t}
    
    CRITICAL: We use positions shifted by 1 period (t-1).
    This prevents LOOK-AHEAD BIAS — a cardinal sin in backtesting.
    The position decided at end of month t trades at month t+1 prices.
    
    Transaction costs:
        TC_t = cost × Σ_i |w_{i,t} - w_{i,t-1}|
    
    Parameters
    ----------
    transaction_cost : float — one-way cost as fraction (0.001 = 10bps)
    """
    # Shift positions by 1: signal at t → position held during t+1
    # This is the no-look-ahead-bias implementation
    lagged_positions = positions.shift(1)
    
    # Gross portfolio return (before costs)
    # Element-wise multiply positions × returns, then sum across assets
    gross_returns = (lagged_positions * returns).sum(axis=1)
    
    # Transaction costs: proportional to portfolio turnover
    # Turnover = sum of absolute position changes
    turnover = lagged_positions.diff().abs().sum(axis=1)
    costs = transaction_cost * turnover
    
    # Net returns
    net_returns = gross_returns - costs
    
    # Equity curve: cumulative product of (1 + r_t)
    # Starting value = 1 (normalised)
    equity_curve = (1 + net_returns).cumprod()
    
    return gross_returns, net_returns, equity_curve, turnover


# ─────────────────────────────────────────────
# SECTION 4: PERFORMANCE METRICS
# ─────────────────────────────────────────────

def compute_performance_metrics(net_returns, equity_curve, risk_free_rate=0.04):
    """
    Compute full suite of institutional performance metrics.
    
    Metrics computed:
    ─ Total Return: cumulative return over full period
    ─ CAGR: Compound Annual Growth Rate
    ─ Sharpe Ratio: risk-adjusted return vs cash
    ─ Sortino Ratio: like Sharpe but penalises only downside vol
    ─ Maximum Drawdown: largest peak-to-trough decline
    ─ Calmar Ratio: CAGR / Max Drawdown
    ─ Win Rate: % of months with positive return
    ─ Avg Win / Avg Loss: payoff ratio
    ─ Monthly VaR (95%): Value at Risk
    """
    monthly_rf = risk_free_rate / 12  # Convert annual RF to monthly
    
    # ── Total Return ──────────────────────────────
    total_return = equity_curve.iloc[-1] - 1
    
    # ── CAGR ──────────────────────────────────────
    n_years = len(net_returns) / 12
    cagr = (equity_curve.iloc[-1]) ** (1 / n_years) - 1
    
    # ── Sharpe Ratio ──────────────────────────────
    # Sharpe = (mean monthly excess return) / (std monthly return) × √12
    # Excess return = return - risk-free rate
    excess_returns = net_returns - monthly_rf
    sharpe = (excess_returns.mean() / net_returns.std()) * np.sqrt(12)
    
    # ── Sortino Ratio ─────────────────────────────
    # Like Sharpe, but denominator uses only DOWNSIDE volatility
    # Downside vol = std of negative returns only
    downside_returns = net_returns[net_returns < monthly_rf]
    downside_vol = downside_returns.std() * np.sqrt(12)
    sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else np.nan
    
    # ── Maximum Drawdown ──────────────────────────
    # MDD = max(peak - trough) / peak over entire history
    rolling_max = equity_curve.cummax()  # Running maximum
    drawdown = (equity_curve - rolling_max) / rolling_max  # Drawdown at each point
    max_drawdown = drawdown.min()  # Most negative value
    
    # ── Calmar Ratio ──────────────────────────────
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    # ── Win Rate ──────────────────────────────────
    wins = (net_returns > 0).sum()
    total_months = len(net_returns)
    win_rate = wins / total_months
    
    # ── Avg Win / Avg Loss ────────────────────────
    avg_win = net_returns[net_returns > 0].mean()
    avg_loss = net_returns[net_returns < 0].mean()
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    
    # ── Value at Risk (95%) ───────────────────────
    # 5th percentile of monthly return distribution
    var_95 = np.percentile(net_returns, 5)
    
    # ── Volatility ────────────────────────────────
    annual_vol = net_returns.std() * np.sqrt(12)
    
    metrics = {
        'total_return': round(total_return * 100, 2),
        'cagr': round(cagr * 100, 2),
        'sharpe_ratio': round(sharpe, 3),
        'sortino_ratio': round(sortino, 3),
        'max_drawdown': round(max_drawdown * 100, 2),
        'calmar_ratio': round(calmar, 3),
        'win_rate': round(win_rate * 100, 2),
        'avg_win_pct': round(avg_win * 100, 3),
        'avg_loss_pct': round(avg_loss * 100, 3),
        'profit_factor': round(profit_factor, 3),
        'var_95_monthly': round(var_95 * 100, 3),
        'annual_vol': round(annual_vol * 100, 2),
        'n_months': total_months,
    }
    
    return metrics, drawdown


# ─────────────────────────────────────────────
# SECTION 5: BENCHMARK COMPARISON
# ─────────────────────────────────────────────

def compute_benchmark(returns):
    """
    Compute simple buy-and-hold SPY benchmark for comparison.
    Equal-weight buy-and-hold across all assets as secondary benchmark.
    """
    # SPY buy-and-hold
    spy_equity = (1 + returns['SPY']).cumprod()
    spy_returns = returns['SPY']
    
    # Equal-weight buy-and-hold
    eq_weight_returns = returns.mean(axis=1)
    eq_equity = (1 + eq_weight_returns).cumprod()
    
    return spy_equity, spy_returns, eq_equity, eq_weight_returns


# ─────────────────────────────────────────────
# SECTION 6: MONTE CARLO SIMULATION
# ─────────────────────────────────────────────

def monte_carlo_simulation(net_returns, n_simulations=500, n_months=60):
    """
    Bootstrap Monte Carlo to estimate strategy robustness.
    
    Method: Block bootstrap (blocks of 6 months) to preserve
    autocorrelation structure of returns. Simple IID bootstrap
    would overstate diversification benefit.
    
    Returns distribution of terminal wealth across simulations.
    """
    block_size = 6  # 6-month blocks
    n_blocks = n_months // block_size
    
    terminal_values = []
    
    for _ in range(n_simulations):
        # Sample blocks randomly with replacement
        simulated_returns = []
        for _ in range(n_blocks):
            # Random start point for block
            start_idx = np.random.randint(0, len(net_returns) - block_size)
            block = net_returns.iloc[start_idx:start_idx + block_size].values
            simulated_returns.extend(block)
        
        # Compute terminal wealth for this simulation path
        path = np.array(simulated_returns[:n_months])
        terminal_wealth = np.prod(1 + path)
        terminal_values.append(terminal_wealth)
    
    return np.array(terminal_values)


# ─────────────────────────────────────────────
# SECTION 7: PAIRS TRADING OVERLAY
# ─────────────────────────────────────────────

def compute_pairs_signal(prices, asset_a='SPY', asset_b='QQQ',
                          lookback=60, entry_z=2.0, exit_z=0.5):
    """
    Cointegration-based pairs trading signal.
    
    Construction:
    1. Rolling OLS: P_A = α + β×P_B + ε  over lookback window
    2. Spread: z_t = P_A - α̂ - β̂×P_B
    3. Z-score: Z_t = (z_t - μ_z) / σ_z
    4. Signal: long spread if Z < -entry_z, short if Z > +entry_z
    
    This is a MARKET-NEUTRAL signal: it bets on relative value,
    not market direction. Combines well with directional TSMOM.
    """
    pa = prices[asset_a]
    pb = prices[asset_b]
    
    spreads = []
    hedge_ratios = []
    
    for i in range(lookback, len(prices)):
        # Rolling window of prices
        pa_window = pa.iloc[i-lookback:i].values
        pb_window = pb.iloc[i-lookback:i].values
        
        # OLS regression: pa = alpha + beta * pb
        # Using matrix form: [1, pb] @ [alpha, beta] = pa
        X = np.column_stack([np.ones(lookback), pb_window])
        # OLS solution: β = (X'X)^{-1} X'y
        beta_hat = np.linalg.lstsq(X, pa_window, rcond=None)[0]
        alpha, beta = beta_hat
        
        # Current spread (out-of-sample point)
        spread = pa.iloc[i] - alpha - beta * pb.iloc[i]
        spreads.append(spread)
        hedge_ratios.append(beta)
    
    spread_series = pd.Series(spreads, index=prices.index[lookback:])
    
    # Z-score the spread
    z_scores = (spread_series - spread_series.rolling(20).mean()) / \
               spread_series.rolling(20).std()
    
    # Generate signal
    pairs_signal = pd.Series(0.0, index=z_scores.index)
    pairs_signal[z_scores < -entry_z] = 1.0   # Long A, short B
    pairs_signal[z_scores > entry_z] = -1.0   # Short A, long B
    pairs_signal[(z_scores.abs() < exit_z)] = 0.0  # Exit
    
    return pairs_signal, z_scores, spread_series


# ─────────────────────────────────────────────
# SECTION 8: MAIN EXECUTION + EXPORT
# ─────────────────────────────────────────────

def run_full_backtest():
    """
    Master function: runs complete backtest and returns all results
    as a JSON-serialisable dictionary for the dashboard.
    """
    print("Building dataset...")
    prices, returns = build_dataset()
    
    print("Generating signals...")
    signals, positions, realised_vol = generate_tsmom_signals(prices, returns)
    
    print("Simulating trades...")
    gross_returns, net_returns, equity_curve, turnover = simulate_trades(
        returns, positions, transaction_cost=0.001
    )
    
    # Remove NaN periods (warmup)
    valid_idx = net_returns.dropna().index
    net_returns_clean = net_returns[valid_idx]
    equity_clean = equity_curve[valid_idx]
    
    print("Computing metrics...")
    metrics, drawdown = compute_performance_metrics(net_returns_clean, equity_clean)
    
    print("Computing benchmark...")
    spy_equity, spy_returns, eq_equity, eq_returns = compute_benchmark(returns)
    
    # Align benchmark to valid strategy index (inner join)
    common_idx = valid_idx.intersection(spy_equity.index)
    spy_clean = spy_equity[common_idx]
    eq_clean = eq_equity[common_idx]
    equity_clean = equity_curve[common_idx]
    net_returns_clean = net_returns[common_idx]
    
    # Rebase benchmarks so they all start at 1.0 on the same date
    spy_clean = spy_clean / spy_clean.iloc[0]
    eq_clean = eq_clean / eq_clean.iloc[0]
    equity_clean = equity_clean / equity_clean.iloc[0]
    
    # Recompute metrics on aligned data
    metrics, drawdown = compute_performance_metrics(net_returns_clean, equity_clean)
    
    # Benchmark metrics
    spy_net = spy_returns[common_idx]
    eq_net = eq_returns[common_idx]
    valid_idx = common_idx
    
    spy_metrics, spy_dd = compute_performance_metrics(spy_net, spy_clean)
    eq_metrics, eq_dd = compute_performance_metrics(eq_net, eq_clean)
    
    print("Running Monte Carlo...")
    mc_results = monte_carlo_simulation(net_returns_clean, n_simulations=500, n_months=60)
    
    print("Computing pairs signal...")
    pairs_signal, z_scores, spread_series = compute_pairs_signal(prices)
    
    # ── Package all results ───────────────────────
    dates_list = [d.strftime('%Y-%m') for d in valid_idx]
    
    # Asset-level signal data (last 24 months for display)
    last_signals = signals.tail(24)
    asset_signal_data = {}
    for col in last_signals.columns:
        asset_signal_data[col] = [
            {'date': d.strftime('%Y-%m'), 'signal': float(v)}
            for d, v in zip(last_signals.index, last_signals[col])
        ]
    
    # Monthly returns for heatmap
    monthly_data = []
    for date, ret in zip(valid_idx, net_returns_clean):
        if not np.isnan(ret):
            monthly_data.append({
                'year': date.year,
                'month': date.month,
                'return': round(ret * 100, 2)
            })
    
    # Position weights over time (last 36 months)
    pos_tail = positions.tail(36)
    position_data = []
    for date, row in pos_tail.iterrows():
        position_data.append({
            'date': date.strftime('%Y-%m'),
            **{col: round(float(val), 4) for col, val in row.items()}
        })
    
    results = {
        'metadata': {
            'strategy': 'Cross-Asset Time-Series Momentum (TSMOM)',
            'universe': list(prices.columns),
            'start_date': dates_list[0],
            'end_date': dates_list[-1],
            'rebalance_freq': 'Monthly',
            'lookback': '12-1 months',
            'vol_target': '15% annualised',
            'transaction_cost': '10bps one-way'
        },
        'metrics': {
            'strategy': metrics,
            'spy': spy_metrics,
            'equal_weight': eq_metrics
        },
        'equity_curves': {
            'dates': dates_list,
            'strategy': [round(float(v), 4) for v in equity_clean],
            'spy': [round(float(v), 4) for v in spy_clean],
            'equal_weight': [round(float(v), 4) for v in eq_clean]
        },
        'drawdown': {
            'dates': dates_list,
            'strategy': [round(float(v) * 100, 2) for v in drawdown[valid_idx]],
            'spy': [round(float(v) * 100, 2) for v in spy_dd[valid_idx]]
        },
        'monthly_returns': monthly_data,
        'positions': position_data,
        'monte_carlo': {
            'terminal_values': [round(float(v), 4) for v in mc_results],
            'percentile_5': round(float(np.percentile(mc_results, 5)), 4),
            'percentile_25': round(float(np.percentile(mc_results, 25)), 4),
            'median': round(float(np.median(mc_results)), 4),
            'percentile_75': round(float(np.percentile(mc_results, 75)), 4),
            'percentile_95': round(float(np.percentile(mc_results, 95)), 4),
        },
        'pairs_trading': {
            'dates': [d.strftime('%Y-%m') for d in z_scores.index],
            'z_scores': [round(float(v), 3) if not np.isnan(v) else 0
                         for v in z_scores],
            'signals': [round(float(v), 1) for v in pairs_signal]
        },
        'asset_signals': asset_signal_data,
        'turnover': {
            'dates': dates_list,
            'values': [round(float(v), 4) if not np.isnan(v) else 0
                       for v in turnover[valid_idx]]
        }
    }
    
    return results


if __name__ == "__main__":
    results = run_full_backtest()
    output_path = '/mnt/user-data/outputs/backtest_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults exported to {output_path}")
    print(f"\nKey Metrics:")
    m = results['metrics']['strategy']
    print(f"  Total Return:    {m['total_return']}%")
    print(f"  CAGR:            {m['cagr']}%")
    print(f"  Sharpe Ratio:    {m['sharpe_ratio']}")
    print(f"  Max Drawdown:    {m['max_drawdown']}%")
    print(f"  Win Rate:        {m['win_rate']}%")
    print(f"  Calmar Ratio:    {m['calmar_ratio']}")
