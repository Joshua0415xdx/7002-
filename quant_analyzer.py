import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend to ensure figures save correctly in headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


class QuantitativeAnalyzer:

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        benchmark: str = "SPY",
        output_dir: str = "figures",
    ) -> None:
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.output_dir = output_dir

        self.data: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None
        self.benchmark_data: pd.Series | None = None
        self.benchmark_returns: pd.Series | None = None

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )

        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------- Utility methods -----------------------------
    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved figure -> {path}")

    # --------------------------------- Data -----------------------------------
    def fetch_data(self) -> bool:
        print("Fetching stock data...")

        try:
            raw_data = yf.download(
                self.symbols,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True,
            )

            if raw_data.columns.nlevels > 1:
                self.data = raw_data["Close"].copy()
            else:
                self.data = raw_data.copy()

            print(f"Successfully fetched data for {len(self.data.columns)} tickers")

            benchmark_raw = yf.download(
                self.benchmark, start=self.start_date, end=self.end_date, progress=False
            )

            if benchmark_raw.columns.nlevels > 1:
                self.benchmark_data = benchmark_raw[("Close", self.benchmark)]
            else:
                self.benchmark_data = benchmark_raw["Close"]

            print(f"Successfully fetched benchmark {self.benchmark} data")

            print("\nData completeness check:")
            for symbol in self.data.columns:
                missing_count = self.data[symbol].isnull().sum()
                total_points = len(self.data[symbol])
                valid_points = total_points - missing_count
                if missing_count > 0:
                    print(f"{symbol}: valid data {valid_points}/{total_points} points")
                else:
                    print(f"{symbol}: complete data ({total_points} points)")

            # Align start date by first fully non-null row across all columns
            if len(self.data.dropna()) > 0:
                valid_data_start = self.data.dropna().index[0]
            else:
                valid_data_start = self.data.index[0]

            self.data = self.data.loc[valid_data_start:].ffill().dropna(how="all")
            self.benchmark_data = self.benchmark_data.loc[valid_data_start:].ffill().dropna()

            # Align dates between stocks and benchmark
            common_dates = self.data.index.intersection(self.benchmark_data.index)
            self.data = self.data.loc[common_dates]
            self.benchmark_data = self.benchmark_data.loc[common_dates]

            print(
                f"\nFinal data range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}"
            )
            print(f"Total {len(self.data)} trading days")

            # Final data quality check
            final_missing = self.data.isnull().sum()
            if final_missing.sum() > 0:
                print("\nFinal missing value summary:")
                for symbol, missing in final_missing.items():
                    if missing > 0:
                        print(f"{symbol}: {missing} missing values")
            else:
                print("\nAll data fully aligned and complete")

        except Exception as e:  # noqa: BLE001
            print(f"Failed to fetch data: {e}")
            import traceback

            traceback.print_exc()
            return False

        return True

    # -------------------------------- Returns ---------------------------------
    def calculate_returns(self) -> None:
        print("\nCalculating daily returns...")
        self.returns = self.data.pct_change().dropna()
        self.benchmark_returns = self.benchmark_data.pct_change().dropna()

        print("Returns summary statistics:")
        print(self.returns.describe())

    # ------------------------------- Correlation ------------------------------
    def correlation_analysis(self) -> pd.DataFrame:
        print("\nPerforming correlation analysis...")

        correlation_matrix = self.returns.corr()

        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8},
            ax=ax,
        )
        ax.set_title("Heatmap of Stock Return Correlations (Pearson)")
        self._save_figure(fig, "correlation_heatmap.png")

        return correlation_matrix

    # ------------------------------- Sharpe ratio -----------------------------
    def sharpe_ratio_analysis(self, risk_free_rate: float = 0.02) -> Tuple[pd.Series, pd.Series, pd.Series]:
        print("\nCalculating Sharpe ratios...")

        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (annual_returns - risk_free_rate) / annual_volatility

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

        # Sharpe ratio bar chart
        colors = ["#2ca02c" if x > 0 else "#d62728" for x in sharpe_ratios]
        sharpe_ratios.sort_values(ascending=False).plot(kind="bar", ax=ax1, color=colors, alpha=0.9)
        ax1.set_title("Sharpe Ratio (Risk-Adjusted Return)")
        ax1.set_ylabel("Sharpe Ratio")
        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # Risk-return scatter
        ax2.scatter(annual_volatility, annual_returns, s=80, alpha=0.8, c="steelblue", edgecolors="white", linewidths=0.8)
        for symbol in self.symbols:
            ax2.annotate(symbol, (annual_volatility[symbol], annual_returns[symbol]), xytext=(5, 4), textcoords="offset points", fontsize=9)
        ax2.set_xlabel("Annualized Volatility")
        ax2.set_ylabel("Annualized Return")
        ax2.set_title("Risk vs. Return")
        ax2.grid(True, alpha=0.3)

        self._save_figure(fig, "sharpe_ratio.png")

        print("Sharpe ratios (descending):")
        print(sharpe_ratios.sort_values(ascending=False))

        return sharpe_ratios, annual_returns, annual_volatility

    # ---------------------------- Portfolio strategies ------------------------
    def portfolio_strategies(self) -> Tuple[pd.DataFrame, pd.Series]:
        print("\nEvaluating portfolio strategies...")

        # 1) Equal-weight portfolio
        equal_weight_returns = self.returns.mean(axis=1)

        # 2) Simplified minimum-variance proxy using inverse average correlation
        correlation_matrix = self.returns.corr()
        avg_correlations = correlation_matrix.mean(axis=1)
        inv_avg_corr = 1 / avg_correlations
        min_var_weights = inv_avg_corr / inv_avg_corr.sum()
        min_var_returns = (self.returns * min_var_weights).sum(axis=1)

        # 3) Momentum strategy (20-day mean return)
        momentum_scores = self.returns.rolling(window=20).mean()
        momentum_weights = momentum_scores.div(momentum_scores.sum(axis=1), axis=0)
        momentum_returns = (self.returns * momentum_weights.shift(1)).sum(axis=1)

        # 4) Benchmark
        benchmark_returns = self.benchmark_returns

        strategies: Dict[str, pd.Series] = {
            "Equal Weight": equal_weight_returns,
            "Min-Var Proxy": min_var_returns,
            "Momentum (20D)": momentum_returns.dropna(),
            f"Benchmark ({self.benchmark})": benchmark_returns,
        }

        cumulative_returns: Dict[str, pd.Series] = {}
        for name, series in strategies.items():
            cumulative_returns[name] = (1 + series).cumprod()

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes.ravel()

        # Cumulative returns
        for name, cum_ret in cumulative_returns.items():
            ax1.plot(cum_ret.index, cum_ret.values, linewidth=2, label=name, alpha=0.9)
        ax1.set_title("Cumulative Returns Comparison")
        ax1.set_ylabel("Cumulative Return")
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax1.grid(True, alpha=0.3)

        # Annualized returns
        annual_rets = {name: ret.mean() * 252 for name, ret in strategies.items()}
        pd.Series(annual_rets).sort_values(ascending=False).plot(kind="bar", ax=ax2, color="skyblue", alpha=0.9)
        ax2.set_title("Annualized Returns")
        ax2.set_ylabel("Annualized Return")
        ax2.tick_params(axis="x", rotation=30)
        ax2.grid(True, alpha=0.3)

        # Annualized volatility
        volatilities = {name: ret.std() * np.sqrt(252) for name, ret in strategies.items()}
        pd.Series(volatilities).sort_values(ascending=True).plot(kind="bar", ax=ax3, color="salmon", alpha=0.9)
        ax3.set_title("Annualized Volatility")
        ax3.set_ylabel("Annualized Volatility")
        ax3.tick_params(axis="x", rotation=30)
        ax3.grid(True, alpha=0.3)

        # Sharpe ratios across strategies (rf = 2%)
        sharpe_by_strategy = {name: (annual_rets[name] - 0.02) / volatilities[name] for name in annual_rets.keys()}
        sharpe_series = pd.Series(sharpe_by_strategy)
        colors = ["#2ca02c" if x > 0 else "#d62728" for x in sharpe_series]
        sharpe_series.plot(kind="bar", ax=ax4, color=colors, alpha=0.9)
        ax4.set_title("Sharpe Ratios (rf=2%)")
        ax4.set_ylabel("Sharpe Ratio")
        ax4.tick_params(axis="x", rotation=30)
        ax4.grid(True, alpha=0.3)

        self._save_figure(fig, "portfolio_strategies.png")

        print("\nPortfolio strategy statistics:")
        strategy_stats = pd.DataFrame(
            {
                "Annualized Return": pd.Series(annual_rets),
                "Annualized Volatility": pd.Series(volatilities),
                "Sharpe Ratio": sharpe_series,
            }
        )
        print(strategy_stats)

        return strategy_stats, min_var_weights

    # ------------------------------- Risk metrics ------------------------------
    def additional_analysis(self) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        print("\nRunning additional risk analyses...")

        # 1) 5% Value at Risk (historical)
        confidence_level = 0.05
        var_5 = self.returns.quantile(confidence_level)

        # 2) Maximum drawdown per asset
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 3) 60-day rolling correlation vs. benchmark
        rolling_corr: Dict[str, pd.Series] = {}
        for symbol in self.symbols:
            if symbol in self.returns.columns:
                rolling_corr[symbol] = self.returns[symbol].rolling(window=60).corr(self.benchmark_returns)
        rolling_corr_df = pd.DataFrame(rolling_corr)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes.ravel()

        # VaR (5%)
        var_5.sort_values(ascending=True).plot(kind="bar", ax=ax1, color="#d62728", alpha=0.9)
        ax1.set_title("5% Historical VaR (per asset)")
        ax1.set_ylabel("VaR")
        ax1.tick_params(axis="x", rotation=30)
        ax1.grid(True, alpha=0.3)

        # Max drawdown
        max_drawdown.sort_values(ascending=True).plot(kind="bar", ax=ax2, color="#8c564b", alpha=0.9)
        ax2.set_title("Maximum Drawdown (per asset)")
        ax2.set_ylabel("Drawdown")
        ax2.tick_params(axis="x", rotation=30)
        ax2.grid(True, alpha=0.3)

        # Drawdown time series
        for symbol in self.symbols:
            if symbol in drawdown.columns:
                ax3.plot(drawdown.index, drawdown[symbol], linewidth=1.5, alpha=0.9, label=symbol)
        ax3.set_title("Drawdown Time Series")
        ax3.set_ylabel("Drawdown")
        ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax3.grid(True, alpha=0.3)

        # 60-day rolling correlation vs. benchmark
        rolling_corr_df.plot(ax=ax4, linewidth=2, alpha=0.9)
        ax4.set_title(f"60-Day Rolling Correlation vs. {self.benchmark}")
        ax4.set_ylabel("Correlation")
        ax4.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax4.grid(True, alpha=0.3)

        self._save_figure(fig, "risk_metrics.png")

        print("5% VaR (ascending):")
        print(var_5.sort_values())
        print("\nMaximum drawdowns (ascending):")
        print(max_drawdown.sort_values())

        return var_5, max_drawdown, rolling_corr_df

    # --------------------------- Comprehensive dashboard ----------------------
    def comprehensive_dashboard(self) -> None:
        print("\nBuilding comprehensive analysis dashboard...")

        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (annual_returns - 0.02) / annual_volatility

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

        # 1) Normalized price trend
        normalized_prices = self.data / self.data.iloc[0]
        normalized_prices.plot(ax=ax1, linewidth=2, alpha=0.9)
        ax1.set_title("Standardized Price Trend (base=1)")
        ax1.set_ylabel("Standardized Price")
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2) Return distribution (box)
        self.returns.plot(kind="box", ax=ax2)
        ax2.set_title("Distribution of Daily Returns (Box Plot)")
        ax2.set_ylabel("Daily Return")
        ax2.tick_params(axis="x", rotation=30, labelsize=8)
        ax2.grid(True, alpha=0.3)

        # 3) Correlation heatmap (simple)
        correlation_matrix = self.returns.corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 7},
            ax=ax3,
        )
        ax3.set_title("Return Correlation")

        # 4) Annualized returns
        annual_returns.sort_values(ascending=False).plot(kind="bar", ax=ax4, color="steelblue", alpha=0.9)
        ax4.set_title("Annualized Return")
        ax4.set_ylabel("Annualized Return")
        ax4.tick_params(axis="x", rotation=30, labelsize=8)
        ax4.grid(True, alpha=0.3)

        # 5) Sharpe ratios
        colors = ["#2ca02c" if x > 0 else "#d62728" for x in sharpe_ratios]
        sharpe_ratios.sort_values(ascending=False).plot(kind="bar", ax=ax5, color=colors, alpha=0.9)
        ax5.set_title("Sharpe Ratio (rf=2%)")
        ax5.set_ylabel("Sharpe Ratio")
        ax5.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax5.tick_params(axis="x", rotation=30, labelsize=8)
        ax5.grid(True, alpha=0.3)

        # 6) Cumulative returns vs. benchmark
        cumulative_returns = (1 + self.returns).cumprod()
        cumulative_returns.plot(ax=ax6, linewidth=2, alpha=0.9)
        benchmark_cum = (1 + self.benchmark_returns).cumprod()
        ax6.plot(
            benchmark_cum.index,
            benchmark_cum.values,
            linewidth=2,
            alpha=0.9,
            label=f"Benchmark ({self.benchmark})",
            color="black",
            linestyle="--",
        )
        ax6.set_title("Cumulative Returns vs. Benchmark")
        ax6.set_ylabel("Cumulative Return")
        ax6.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        ax6.grid(True, alpha=0.3)

        fig.suptitle("Stock Quantitative Analysis Dashboard", fontsize=16)
        self._save_figure(fig, "dashboard.png")


def main() -> None:
    print("=" * 80)
    print("Stock Quantitative Analysis System")
    print("Tickers: OPEN, TLT, TQQQ, TSLA, META, NVDA")
    print("Date range: 2020-09-15 to 2025-09-15")
    print("=" * 80)

    symbols = ["OPEN", "TLT", "TQQQ", "TSLA", "META", "NVDA"]
    start_date = "2020-09-15"
    end_date = "2025-09-15"

    analyzer = QuantitativeAnalyzer(symbols, start_date, end_date)

    try:
        if not analyzer.fetch_data():
            print("Data fetch failed. Exiting.")
            return

        analyzer.calculate_returns()

        _ = analyzer.correlation_analysis()
        _ = analyzer.sharpe_ratio_analysis()
        _ = analyzer.portfolio_strategies()
        _ = analyzer.additional_analysis()
        analyzer.comprehensive_dashboard()

        print("\n" + "=" * 80)
        print(f"Analysis complete! All figures saved under: {analyzer.output_dir}")
        print("=" * 80)

    except Exception as e:  # noqa: BLE001
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

