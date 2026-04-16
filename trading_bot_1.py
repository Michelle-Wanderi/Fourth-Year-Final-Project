#!/usr/bin/env python3
"""

Run:
    pip install yfinance pandas pandas-ta scikit-learn backtrader matplotlib seaborn
    python trading_bot.py
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════
CONFIG = {
    "symbols":        ["BTC-USD", "ETH-USD"],
    "start_date":     "2018-01-01",
    "end_date":       "2024-12-31",
    "train_ratio":    0.80,          # 80% train, 20% test (chronological)
    "initial_capital": 10_000,       # USD
    "position_size":  0.10,          # 10% of capital per trade
    "stop_loss":      0.03,          # 3% stop-loss per trade
    "transaction_cost": 0.001,       # 0.1% per side
    "slippage":       0.0005,        # 0.05% adverse fill assumption
    "n_estimators":   100,           # Random Forest trees
    "random_state":   42,
    "use_walk_forward": True,        # robust validation over single split
    "wf_train_size":  700,           # initial train days
    "wf_test_size":   120,           # days per walk-forward fold
    "wf_step_size":   120,           # step between folds
    "paper_trading_mode": False,     # set True to run paper trading cycle
    "paper_state_file": "paper_state.json",
    "paper_trade_log": "paper_trades.csv",
    "features": [
        "RSI_14", "MACD", "MACD_signal",
        "SMA_20", "SMA_50",
        "BB_upper", "BB_lower", "BB_mid",
        "return_1d", "return_3d", "Volume"
    ]
}

# ═══════════════════════════════════════════════════════
#  STEP 1: DATA COLLECTION
# ═══════════════════════════════════════════════════════
def collect_data(symbol: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance (Binance API alternative).
    Uses yfinance — no API key required.
    """
    print(f"  Downloading {symbol} data ({CONFIG['start_date']} → {CONFIG['end_date']})...")
    df = yf.download(
        symbol,
        start=CONFIG["start_date"],
        end=CONFIG["end_date"],
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol}. Check your internet connection.")
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    print(f"  ✓ {symbol}: {len(df)} trading days | "
          f"${df['Close'].min():.0f} – ${df['Close'].max():.0f}")
    return df


# ═══════════════════════════════════════════════════════
#  STEP 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and create ML features.
    
    Indicators used (per proposal §3.4):
    - RSI-14        : momentum / overbought-oversold
    - MACD          : trend direction & momentum shifts
    - SMA-20, SMA-50: moving average baseline features
    - Bollinger Bands: volatility & reversal signals
    - Lagged returns : short-term momentum (1d, 3d)
    
    Target: 1 if next-day close > today's close (Buy), else 0 (Hold/Sell)
    """
    df = df.copy()

    # RSI
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # MACD
    macd = ta.macd(df["Close"])
    df["MACD"]        = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]

    # Simple Moving Averages
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)

    # Bollinger Bands
    bb = ta.bbands(df["Close"], length=20)
    # Column names vary by pandas_ta version — handle both
    bb_cols = bb.columns.tolist()
    upper_col = [c for c in bb_cols if c.startswith("BBU")][0]
    lower_col = [c for c in bb_cols if c.startswith("BBL")][0]
    mid_col   = [c for c in bb_cols if c.startswith("BBM")][0]
    df["BB_upper"] = bb[upper_col]
    df["BB_lower"] = bb[lower_col]
    df["BB_mid"]   = bb[mid_col]

    # Lagged returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_3d"] = df["Close"].pct_change(3)

    # Binary target: 1 = price goes up tomorrow
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df


# ═══════════════════════════════════════════════════════
#  STEP 3: MODEL TRAINING — RANDOM FOREST
# ═══════════════════════════════════════════════════════
def train_random_forest(df: pd.DataFrame, symbol: str):
    """
    Train Random Forest on 80% of data (chronological split).
    Returns: (model, scaler, test_df with predictions)
    """
    features = CONFIG["features"]
    split    = int(len(df) * CONFIG["train_ratio"])
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    print(f"\n  Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} days)")
    print(f"  Test:  {test.index[0].date()}  → {test.index[-1].date()}  ({len(test)} days)")

    # Min-Max normalisation (no lookahead — fit on train only)
    scaler    = MinMaxScaler()
    X_train_s = scaler.fit_transform(train[features])
    X_test_s  = scaler.transform(test[features])

    rf = RandomForestClassifier(
        n_estimators=CONFIG["n_estimators"],
        random_state=CONFIG["random_state"],
        n_jobs=-1
    )
    rf.fit(X_train_s, train["target"])
    preds = rf.predict(X_test_s)
    test["signal"] = preds

    acc = accuracy_score(test["target"], preds)
    print(f"\n  ── Random Forest: {symbol} ──")
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(test["target"], preds,
                                target_names=["Hold/Sell", "Buy"], digits=3))

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("  Feature importances:")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:15s} {bar} {imp:.4f}")

    return rf, scaler, test


def walk_forward_random_forest(df: pd.DataFrame, symbol: str):
    """
    Expanding-window walk-forward evaluation.
    Returns: (predictions_df, fold_summaries)
    """
    features = CONFIG["features"]
    train_size = CONFIG["wf_train_size"]
    test_size = CONFIG["wf_test_size"]
    step = CONFIG["wf_step_size"]

    if len(df) < (train_size + test_size):
        print("  [WF] Not enough rows, falling back to single split.")
        _, _, test = train_random_forest(df, symbol)
        return test, []

    fold_predictions = []
    fold_summaries = []
    fold = 1

    for start in range(train_size, len(df) - test_size + 1, step):
        train = df.iloc[:start].copy()
        test = df.iloc[start:start + test_size].copy()

        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(train[features])
        X_test_s = scaler.transform(test[features])

        rf = RandomForestClassifier(
            n_estimators=CONFIG["n_estimators"],
            random_state=CONFIG["random_state"],
            n_jobs=-1
        )
        rf.fit(X_train_s, train["target"])
        preds = rf.predict(X_test_s)
        test["signal"] = preds
        test["wf_fold"] = fold
        fold_predictions.append(test)

        acc = accuracy_score(test["target"], preds)
        fold_summaries.append({
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "acc": acc,
            "start": str(test.index[0].date()),
            "end": str(test.index[-1].date()),
        })
        print(f"  [WF] Fold {fold:02d}: {test.index[0].date()} -> "
              f"{test.index[-1].date()} | Acc={acc:.4f}")
        fold += 1

    combined = pd.concat(fold_predictions).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined, fold_summaries


# ═══════════════════════════════════════════════════════
#  BASELINE — MOVING AVERAGE CROSSOVER (SMA 20/50)
# ═══════════════════════════════════════════════════════
def moving_average_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based baseline (§3.5):
    Buy  when SMA-20 crosses above SMA-50
    Sell when SMA-20 crosses below SMA-50
    """
    test = df.iloc[int(len(df) * CONFIG["train_ratio"]):].copy()
    test["signal"] = (test["SMA_20"] > test["SMA_50"]).astype(int)
    return test


def buy_and_hold_benchmark(test: pd.DataFrame, label: str) -> dict:
    """
    Buy-and-hold benchmark over same test window.
    """
    if test.empty:
        raise ValueError("Benchmark received empty test data.")
    start_price = float(test["Close"].iloc[0])
    qty = CONFIG["initial_capital"] / start_price
    equity = qty * test["Close"]
    return compute_metrics(equity, [], label)


def compute_metrics(equity_s: pd.Series, trades: list, label: str) -> dict:
    """
    Compute return and risk metrics from an equity curve.
    """
    final_capital = float(equity_s.iloc[-1])
    total_return = (final_capital - CONFIG["initial_capital"]) / CONFIG["initial_capital"] * 100
    daily_ret = equity_s.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)

    downside = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / downside.std() * np.sqrt(252)
               if len(downside) > 1 and downside.std() > 0 else 0.0)

    roll_max = equity_s.cummax()
    max_dd = ((equity_s - roll_max) / roll_max).min() * 100

    years = max((equity_s.index[-1] - equity_s.index[0]).days / 365.25, 1 / 365.25)
    cagr = ((final_capital / CONFIG["initial_capital"]) ** (1 / years) - 1) * 100
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    completed = [t for t in trades if t.get("type") in ("SELL", "STOP-LOSS")]
    wins = [t for t in completed if t["pnl"] > 0]
    losses = [t for t in completed if t["pnl"] <= 0]
    win_rate = len(wins) / len(completed) * 100 if completed else 0.0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
    expectancy = (sum(t["pnl"] for t in completed) / len(completed)) if completed else 0.0

    return {
        "label": label,
        "final_capital": final_capital,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "n_trades": len(completed),
        "equity": equity_s,
        "trades": pd.DataFrame(trades),
    }


# ═══════════════════════════════════════════════════════
#  STEP 4: BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════
def backtest(test: pd.DataFrame, label: str) -> dict:
    """
    Simulate trading on the test set.
    - 10% position sizing per trade (§3.8)
    - 3% stop-loss per trade (§3.8)
    - 0.1% transaction cost per trade
    
    Returns a dict of performance metrics + equity curve.
    """
    capital    = CONFIG["initial_capital"]
    position   = 0.0          # units held
    entry_price = 0.0
    trades     = []
    equity     = []

    for date, row in test.iterrows():
        price = float(row["Close"])
        sig   = int(row["signal"])
        fee = CONFIG["transaction_cost"]
        slip = CONFIG["slippage"]

        # ── Stop-loss check ─────────────────────────────
        if position > 0 and price < entry_price * (1 - CONFIG["stop_loss"]):
            stop_fill = price * (1 - slip)  # adverse fill for sell
            proceeds = position * stop_fill * (1 - fee)
            pnl      = proceeds - (position * entry_price)
            capital += proceeds
            trades.append({"date": date, "type": "STOP-LOSS",
                           "price": stop_fill, "pnl": pnl})
            position = 0.0

        # ── Buy ─────────────────────────────────────────
        if sig == 1 and position == 0:
            invest   = capital * CONFIG["position_size"]
            buy_fill = price * (1 + slip)  # adverse fill for buy
            position = (invest * (1 - fee)) / buy_fill
            capital -= invest
            entry_price = buy_fill
            trades.append({"date": date, "type": "BUY",
                           "price": buy_fill, "pnl": 0})

        # ── Sell ────────────────────────────────────────
        elif sig == 0 and position > 0:
            sell_fill = price * (1 - slip)
            proceeds = position * sell_fill * (1 - fee)
            pnl      = proceeds - (position * entry_price)
            capital += proceeds
            trades.append({"date": date, "type": "SELL",
                           "price": sell_fill, "pnl": pnl})
            position = 0.0

        equity.append(capital + position * price)

    # Close any remaining position at last price
    if position > 0:
        last_price = float(test["Close"].iloc[-1])
        sell_fill = last_price * (1 - CONFIG["slippage"])
        capital   += position * sell_fill * (1 - CONFIG["transaction_cost"])
        equity[-1] = capital

    equity_s = pd.Series(equity, index=test.index)
    return compute_metrics(equity_s, trades, label)


# ═══════════════════════════════════════════════════════
#  STEP 5: PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════
def generate_report(results: list, dfs: dict):
    """
    Produce a dark-themed performance dashboard:
    - Equity curves (RF vs MA, BTC & ETH)
    - Metrics bar charts
    - Summary table
    """
    BG    = "#0D1117"; PANEL = "#161B22"; GOLD  = "#F0B429"
    TEAL  = "#00D4AA"; RED   = "#FF4757"; BLUE  = "#3D8EF7"
    GREY  = "#8B949E"; WHITE = "#E6EDF3"

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "axes.edgecolor": "#30363D", "axes.labelcolor": WHITE,
        "xtick.color": GREY, "ytick.color": GREY,
        "text.color": WHITE, "grid.color": "#21262D",
        "grid.linestyle": "--", "grid.linewidth": 0.5,
        "font.family": "monospace",
    })

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                             hspace=0.48, wspace=0.32,
                             left=0.07, right=0.97,
                             top=0.91, bottom=0.07)

    fig.text(0.5, 0.965,
             "ML-DRIVEN CRYPTOCURRENCY TRADING BOT",
             ha="center", fontsize=18, fontweight="bold",
             color=GOLD, fontfamily="monospace")
    fig.text(0.5, 0.948,
             "Random Forest vs MA Crossover  ·  Michelle Wangari Wanderi  ·  JKUAT",
             ha="center", fontsize=10, color=GREY, fontfamily="monospace")

    rf_btc = next(r for r in results if "BTC" in r["label"] and "Forest" in r["label"])
    ma_btc = next(r for r in results if "BTC" in r["label"] and "MA"     in r["label"])
    rf_eth = next(r for r in results if "ETH" in r["label"] and "Forest" in r["label"])
    ma_eth = next(r for r in results if "ETH" in r["label"] and "MA"     in r["label"])

    def equity_ax(ax, rf, ma, colour, title):
        ax.plot(rf["equity"], color=colour, lw=2,   label="Random Forest")
        ax.plot(ma["equity"], color=GREY,   lw=1.5, ls="--", label="MA Crossover")
        ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True)

    equity_ax(fig.add_subplot(gs[0, 0]),
              rf_btc, ma_btc, GOLD, "BTC/USD — Equity Curve")
    equity_ax(fig.add_subplot(gs[0, 1]),
              rf_eth, ma_eth, TEAL, "ETH/USD — Equity Curve")

    def metric_ax(ax, rf, ma, colour, title):
        labels = ["Return %", "Sharpe×10", "Win Rate %"]
        rf_v   = [rf["total_return"], rf["sharpe"]*10, rf["win_rate"]]
        ma_v   = [ma["total_return"], ma["sharpe"]*10, ma["win_rate"]]
        x, w   = np.arange(3), 0.35
        ax.bar(x - w/2, rf_v, w, color=colour, alpha=0.85, label="RF")
        ax.bar(x + w/2, ma_v, w, color=GREY,   alpha=0.85, label="MA")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, axis="y")

    metric_ax(fig.add_subplot(gs[1, 0]),
              rf_btc, ma_btc, GOLD, "BTC — Performance Metrics")
    metric_ax(fig.add_subplot(gs[1, 1]),
              rf_eth, ma_eth, TEAL, "ETH — Performance Metrics")

    ax_t = fig.add_subplot(gs[2, :])
    ax_t.axis("off")
    cols = ["Strategy", "Return %", "CAGR %", "Sharpe", "Sortino", "Max DD %", "ProfitFactor", "Trades"]
    rows = [[r["label"],
             f'{r["total_return"]:.2f}%',
             f'{r["cagr"]:.2f}%',
             f'{r["sharpe"]:.3f}',
             f'{r["sortino"]:.3f}',
             f'{r["max_drawdown"]:.2f}%',
             f'{r["profit_factor"]:.2f}' if pd.notna(r["profit_factor"]) else "N/A",
             str(r["n_trades"])] for r in results]
    tbl  = ax_t.table(cellText=rows, colLabels=cols,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#21262D" if r == 0 else PANEL)
        cell.set_text_props(
            color=GOLD if r == 0 else WHITE,
            fontweight="bold" if r == 0 else "normal")
        cell.set_edgecolor("#30363D")
    ax_t.set_title("BACKTEST SUMMARY", color=WHITE,
                    fontsize=12, fontweight="bold", pad=20)

    out = "backtest_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"\n  ✓ Report saved → {out}")
    plt.close()


def _load_paper_state() -> dict:
    if os.path.exists(CONFIG["paper_state_file"]):
        with open(CONFIG["paper_state_file"], "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "cash": CONFIG["initial_capital"],
        "positions": {},
        "updated_at": datetime.utcnow().isoformat(),
    }


def _save_paper_state(state: dict):
    state["updated_at"] = datetime.utcnow().isoformat()
    with open(CONFIG["paper_state_file"], "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _append_paper_trade(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(CONFIG["paper_trade_log"]):
        df.to_csv(CONFIG["paper_trade_log"], mode="a", header=False, index=False)
    else:
        df.to_csv(CONFIG["paper_trade_log"], index=False)


def paper_trade_cycle(df: pd.DataFrame, symbol: str, model, scaler):
    """
    One paper-trade cycle based on latest signal.
    Schedule this script daily/periodically when paper_trading_mode=True.
    """
    state = _load_paper_state()
    short = symbol.split("-")[0]
    latest = df.iloc[-1]
    features = CONFIG["features"]
    X = scaler.transform(df[features].iloc[[-1]])
    signal = int(model.predict(X)[0])
    price = float(latest["Close"])
    fee = CONFIG["transaction_cost"]
    slip = CONFIG["slippage"]

    pos = state["positions"].get(short, {"qty": 0.0, "entry_price": 0.0})
    qty = float(pos["qty"])

    action = "HOLD"
    if signal == 1 and qty == 0:
        invest = float(state["cash"]) * CONFIG["position_size"]
        buy_fill = price * (1 + slip)
        qty = (invest * (1 - fee)) / buy_fill
        state["cash"] -= invest
        state["positions"][short] = {"qty": qty, "entry_price": buy_fill}
        action = "BUY"
        _append_paper_trade({
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "action": action,
            "price": buy_fill,
            "qty": qty,
            "cash_after": state["cash"],
        })
    elif signal == 0 and qty > 0:
        sell_fill = price * (1 - slip)
        proceeds = qty * sell_fill * (1 - fee)
        state["cash"] += proceeds
        state["positions"][short] = {"qty": 0.0, "entry_price": 0.0}
        action = "SELL"
        _append_paper_trade({
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "action": action,
            "price": sell_fill,
            "qty": qty,
            "cash_after": state["cash"],
        })

    _save_paper_state(state)
    print(f"  [PAPER] {symbol}: signal={signal} action={action} close={price:.2f}")


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  ML CRYPTO TRADING BOT — RBI Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = []
    dfs         = {}

    for symbol in CONFIG["symbols"]:
        short = symbol.split("-")[0]
        print(f"\n{'─'*60}")
        print(f"  PROCESSING: {symbol}")
        print(f"{'─'*60}")

        # 1. Collect
        df_raw = collect_data(symbol)

        # 2. Features
        df = engineer_features(df_raw)
        dfs[short] = df

        if CONFIG["use_walk_forward"]:
            print("\n  Running walk-forward validation...")
            test_rf, wf_folds = walk_forward_random_forest(df, short)
            print(f"  [WF] Total folds: {len(wf_folds)}")

            # train final model on full history for paper-trading predictions
            scaler = MinMaxScaler()
            X_all = scaler.fit_transform(df[CONFIG["features"]])
            rf = RandomForestClassifier(
                n_estimators=CONFIG["n_estimators"],
                random_state=CONFIG["random_state"],
                n_jobs=-1
            )
            rf.fit(X_all, df["target"])
        else:
            # 3. Train RF
            rf, scaler, test_rf = train_random_forest(df, short)

        # 4. MA baseline
        if CONFIG["use_walk_forward"]:
            test_ma = test_rf.copy()
            test_ma["signal"] = (test_ma["SMA_20"] > test_ma["SMA_50"]).astype(int)
        else:
            test_ma = moving_average_baseline(df)

        # 5. Backtest both
        res_rf = backtest(test_rf, f"{short} — Random Forest")
        res_ma = backtest(test_ma, f"{short} — MA Crossover")
        res_bh = buy_and_hold_benchmark(test_rf, f"{short} — Buy & Hold")
        all_results.extend([res_rf, res_ma, res_bh])

        if CONFIG["paper_trading_mode"]:
            paper_trade_cycle(df, symbol, rf, scaler)

        # Print quick summary
        print(f"\n  ┌─ RF  : Return={res_rf['total_return']:+.2f}%  "
              f"Sharpe={res_rf['sharpe']:.3f}  "
              f"Sortino={res_rf['sortino']:.3f}  "
              f"MaxDD={res_rf['max_drawdown']:.2f}%  "
              f"Win={res_rf['win_rate']:.1f}%  Trades={res_rf['n_trades']}")
        print(f"  └─ MA  : Return={res_ma['total_return']:+.2f}%  "
              f"Sharpe={res_ma['sharpe']:.3f}  "
              f"MaxDD={res_ma['max_drawdown']:.2f}%  "
              f"Win={res_ma['win_rate']:.1f}%  Trades={res_ma['n_trades']}")
        print(f"     BH  : Return={res_bh['total_return']:+.2f}%  "
              f"CAGR={res_bh['cagr']:.2f}%  "
              f"Sharpe={res_bh['sharpe']:.3f}")

    # 6. Report
    print(f"\n{'─'*60}")
    print("  GENERATING PERFORMANCE REPORT ...")
    generate_report(all_results, dfs)
    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
