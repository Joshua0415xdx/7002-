import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


class QuantitativeAnalyzer:

    def __init__(self, symbols, start_date, end_date, benchmark='SPY', output_dir=None, figure_dpi=300):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.data = None
        self.returns = None
        self.benchmark_data = None
        self.benchmark_returns = None

        # Figure output setup
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dpi = figure_dpi

        # Global plot style
        sns.set_theme(style="whitegrid", context="talk", palette="tab10")
        plt.rcParams.update({
            "figure.autolayout": True,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

    def _savefig(self, fig, filename):
        path = self.output_dir / filename
        fig.savefig(path.as_posix(), dpi=self.figure_dpi, bbox_inches='tight')
        return path

    def fetch_data(self):
        print('Fetching stock price data...')

        try:
            raw_data = yf.download(self.symbols, start=self.start_date, end=self.end_date,
                                   progress=False, auto_adjust=True)

            if raw_data.columns.nlevels > 1:
                self.data = raw_data['Close'].copy()
            else:
                # Single ticker path: keep as DataFrame for consistency
                if 'Close' in raw_data.columns:
                    close_series = raw_data['Close'].copy()
                    self.data = close_series.to_frame(name=self.symbols[0])
                else:
                    self.data = raw_data.copy()

            print(f"Successfully fetched {len(self.data.columns)} tickers")

            benchmark_raw = yf.download(self.benchmark, start=self.start_date, end=self.end_date,
                                        progress=False, auto_adjust=True)

            if hasattr(benchmark_raw, 'columns') and getattr(benchmark_raw.columns, 'nlevels', 1) > 1:
                self.benchmark_data = benchmark_raw[('Close', self.benchmark)]
            else:
                self.benchmark_data = benchmark_raw['Close']

            print(f"Successfully fetched benchmark index {self.benchmark} data")

            print("\nData integrity check:")
            for symbol in self.data.columns:
                missing_count = self.data[symbol].isnull().sum()
                total_points = len(self.data[symbol])
                valid_points = total_points - missing_count

                if missing_count > 0:
                    print(f"{symbol}: valid data {valid_points}/{total_points} points")
                else:
                    print(f"{symbol}: complete data ({total_points} points)")

            # Align start where at least one column is valid
            dropped = self.data.dropna()
            if len(dropped) > 0:
                valid_data_start = dropped.index[0]
            else:
                valid_data_start = self.data.index[0]

            self.data = self.data.loc[valid_data_start:].ffill().dropna(how='all')
            self.benchmark_data = self.benchmark_data.loc[valid_data_start:].ffill().dropna()

            # Align dates between stocks and benchmark
            common_dates = self.data.index.intersection(self.benchmark_data.index)
            self.data = self.data.loc[common_dates]
            self.benchmark_data = self.benchmark_data.loc[common_dates]

            print(
                f"\nFinal data date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
            print(f"Total {len(self.data)} trading days")

            # Final missing check
            final_missing = self.data.isnull().sum()
            if final_missing.sum() > 0:
                print("\nFinal missing data:")
                for symbol, missing in final_missing.items():
                    if missing > 0:
                        print(f"{symbol}: {missing} missing values")
            else:
                print("\nAll series are aligned with no missing values")

        except Exception as e:
            print(f"Data fetch failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    def calculate_returns(self):
        """Compute daily returns"""
        print("\nCalculating returns...")

        # Stock daily returns
        self.returns = self.data.pct_change().dropna()

        # Benchmark daily returns
        self.benchmark_returns = self.benchmark_data.pct_change().dropna()

        print("Return statistics summary:")
        print(self.returns.describe())

    def correlation_analysis(self):
        """Correlation analysis - Pearson correlation coefficients"""
        print("\nPerforming correlation analysis...")

        corr = self.returns.corr()

        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            annot_kws={"size": 9},
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            linewidths=0.5,
            linecolor='white',
            cbar_kws={"shrink": .8}
        )
        ax.set_title('Heatmap of stock return correlations (Pearson)')
        fig.tight_layout()
        self._savefig(fig, 'correlation_heatmap.png')
        plt.show()

        return corr

    def sharpe_ratio_analysis(self, risk_free_rate=0.02):
        """Compute Sharpe ratio - risk-adjusted returns"""
        print("\nComputing Sharpe ratios...")

        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (annual_returns - risk_free_rate) / annual_volatility

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Sharpe ratio bar
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in sharpe_ratios]
        sharpe_ratios.sort_values(ascending=False).plot(kind='bar', ax=ax1, color=colors, alpha=0.9)
        ax1.set_title('Sharpe Ratio (risk-adjusted)')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Risk-return scatter
        ax2.scatter(annual_volatility, annual_returns, s=110, alpha=0.8, c='steelblue', edgecolor='white', linewidth=1)
        for symbol in annual_returns.index:
            ax2.annotate(symbol, (annual_volatility.loc[symbol], annual_returns.loc[symbol]),
                         xytext=(6, 6), textcoords='offset points', fontsize=10)
        ax2.set_xlabel('Annualized Volatility')
        ax2.set_ylabel('Annualized Return')
        ax2.set_title('Risk-Return Scatter')
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        self._savefig(fig, 'sharpe_ratio_and_risk_return.png')
        plt.show()

        print("Sharpe ratio ranking:")
        print(sharpe_ratios.sort_values(ascending=False))

        return sharpe_ratios, annual_returns, annual_volatility

    def portfolio_strategies(self):
        """Portfolio strategy simulation"""
        print("\nPerforming portfolio strategy analysis...")

        # 1. Equal-weight portfolio
        equal_weight_returns = self.returns.mean(axis=1)

        # 2. Simplified min-variance via inverse of average correlations
        corr = self.returns.corr()
        avg_corr = corr.mean(axis=1)
        inv_avg = (1 / avg_corr).replace([np.inf, -np.inf], np.nan).fillna(0)
        min_var_weights = inv_avg / inv_avg.sum()
        min_var_returns = (self.returns * min_var_weights).sum(axis=1)

        # 3. Momentum strategy (20-day mean returns)
        momentum_scores = self.returns.rolling(window=20).mean()
        momentum_weights = momentum_scores.div(momentum_scores.sum(axis=1), axis=0)
        momentum_weights = momentum_weights.replace([np.inf, -np.inf], np.nan).fillna(0)
        momentum_returns = (self.returns * momentum_weights.shift(1)).sum(axis=1)

        # 4. Benchmark returns
        benchmark_returns = self.benchmark_returns

        # Cumulative returns
        strategies = {
            'Equal-Weight Portfolio': equal_weight_returns,
            'Min-Variance Portfolio': min_var_returns,
            'Momentum Strategy': momentum_returns.dropna(),
            f'{self.benchmark} Benchmark': benchmark_returns
        }

        cumulative_returns = {name: (1 + ret).cumprod() for name, ret in strategies.items()}

        # Plot comparisons
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Cumulative return curves
        for name, cum_ret in cumulative_returns.items():
            ax1.plot(cum_ret.index, cum_ret.values, linewidth=2, label=name, alpha=0.9)
        ax1.set_title('Cumulative Return Comparison')
        ax1.set_ylabel('Cumulative Return (×)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Annualized returns
        annual_rets = {name: ret.mean() * 252 for name, ret in strategies.items()}
        pd.Series(annual_rets).sort_values(ascending=False).plot(kind='bar', ax=ax2, color='skyblue', alpha=0.9)
        ax2.set_title('Annualized Return by Strategy')
        ax2.set_ylabel('Annualized Return')
        ax2.tick_params(axis='x', rotation=20)
        ax2.grid(True, alpha=0.3)

        # Volatility
        volatilities = {name: ret.std() * np.sqrt(252) for name, ret in strategies.items()}
        pd.Series(volatilities).sort_values(ascending=False).plot(kind='bar', ax=ax3, color='salmon', alpha=0.9)
        ax3.set_title('Annualized Volatility by Strategy')
        ax3.set_ylabel('Annualized Volatility')
        ax3.tick_params(axis='x', rotation=20)
        ax3.grid(True, alpha=0.3)

        # Sharpe ratios
        sharpe = {name: (annual_rets[name] - 0.02) / volatilities[name] for name in annual_rets.keys()}
        sharpe_series = pd.Series(sharpe).sort_values(ascending=False)
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in sharpe_series]
        sharpe_series.plot(kind='bar', ax=ax4, color=colors, alpha=0.9)
        ax4.set_title('Sharpe Ratio by Strategy')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.tick_params(axis='x', rotation=20)
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        self._savefig(fig, 'portfolio_strategy_comparison.png')
        plt.show()

        print("\nPortfolio strategy stats:")
        strategy_stats = pd.DataFrame({
            'Annualized Return': pd.Series(annual_rets),
            'Annualized Volatility': pd.Series(volatilities),
            'Sharpe Ratio': pd.Series(sharpe)
        })
        print(strategy_stats)

        return strategy_stats, min_var_weights

    def additional_analysis(self):
        """Additional quantitative analyses"""
        print("\nPerforming additional quantitative analyses...")

        # 1. 5% VaR (Value at Risk)
        confidence_level = 0.05
        var_5 = self.returns.quantile(confidence_level)

        # 2. Max drawdown per asset
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 3. 60-day rolling correlation vs benchmark
        rolling_corr = {}
        for symbol in self.symbols:
            if symbol in self.returns.columns:
                rolling_corr[symbol] = self.returns[symbol].rolling(window=60).corr(self.benchmark_returns)
        rolling_corr_df = pd.DataFrame(rolling_corr)

        # Plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # VaR
        var_5.sort_values(ascending=True).plot(kind='bar', ax=ax1, color='#d62728', alpha=0.9)
        ax1.set_title('5% VaR (Value at Risk)')
        ax1.set_ylabel('VaR')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Max drawdown
        max_drawdown.sort_values(ascending=True).plot(kind='bar', ax=ax2, color='#8c564b', alpha=0.9)
        ax2.set_title('Maximum Drawdown')
        ax2.set_ylabel('Max Drawdown')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # Drawdown time series
        for symbol in self.symbols:
            if symbol in drawdown.columns:
                ax3.plot(drawdown.index, drawdown[symbol], linewidth=1.6, alpha=0.9, label=symbol)
        ax3.set_title('Drawdown Time Series')
        ax3.set_ylabel('Drawdown')
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)

        # 60-day rolling correlation
        rolling_corr_df.plot(ax=ax4, linewidth=2, alpha=0.9)
        ax4.set_title(f'60-day Rolling Correlation (vs {self.benchmark})')
        ax4.set_ylabel('Correlation')
        ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        self._savefig(fig, 'risk_and_correlation.png')
        plt.show()

        print("5% VaR:")
        print(var_5.sort_values())
        print("\nMaximum drawdown:")
        print(max_drawdown.sort_values())

        return var_5, max_drawdown, rolling_corr_df

    def comprehensive_dashboard(self):
        """Create a comprehensive analysis dashboard"""
        print("\nCreating comprehensive analysis dashboard...")

        # Key metrics
        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (annual_returns - 0.02) / annual_volatility

        # Dashboard layout (2x3 to avoid empty cells)
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        # 1) Normalized price trend
        normalized_prices = self.data / self.data.iloc[0]
        normalized_prices.plot(ax=ax1, linewidth=2, alpha=0.9)
        ax1.set_title('Standardized Price Trend (base=1)')
        ax1.set_ylabel('Standardized Price')
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2) Return distribution (box)
        self.returns.plot(kind='box', ax=ax2)
        ax2.set_title('Distribution of Daily Returns')
        ax2.set_ylabel('Daily Return')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.grid(True, alpha=0.3)

        # 3) Correlation heatmap (compact)
        corr = self.returns.corr()
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f',
                    cbar_kws={"shrink": .8}, ax=ax3, annot_kws={"size": 8}, linewidths=0.3, linecolor='white')
        ax3.set_title('Return Correlation')

        # 4) Annualized return
        annual_returns.sort_values(ascending=False).plot(kind='bar', ax=ax4, color='steelblue', alpha=0.9)
        ax4.set_title('Annualized Return')
        ax4.set_ylabel('Annualized Return')
        ax4.tick_params(axis='x', rotation=45, labelsize=8)
        ax4.grid(True, alpha=0.3)

        # 5) Sharpe ratio
        sr_sorted = sharpe_ratios.sort_values(ascending=False)
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in sr_sorted]
        sr_sorted.plot(kind='bar', ax=ax5, color=colors, alpha=0.9)
        ax5.set_title('Sharpe Ratio')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.tick_params(axis='x', rotation=45, labelsize=8)
        ax5.grid(True, alpha=0.3)

        # 6) Cumulative returns vs benchmark
        cum_rets = (1 + self.returns).cumprod()
        cum_rets.plot(ax=ax6, linewidth=2, alpha=0.9)
        benchmark_cum = (1 + self.benchmark_returns).cumprod()
        ax6.plot(benchmark_cum.index, benchmark_cum.values, linewidth=2, alpha=0.9,
                 label=f'{self.benchmark} Benchmark', color='black', linestyle='--')
        ax6.set_title('Cumulative Return Comparison')
        ax6.set_ylabel('Cumulative Return (×)')
        ax6.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)

        fig.suptitle('Equities Quantitative Analysis Dashboard', fontsize=20, y=1.02)
        fig.tight_layout()
        self._savefig(fig, 'comprehensive_dashboard.png')
        plt.show()


def main():
    """Main function - run the full quantitative analysis pipeline"""
    print("=" * 80)
    print("Stock Quantitative Analysis System")
    print("Analyzing tickers: OPEN, TLT, TQQQ, TSLA, META, NVDA")
    print("Date range: 2020-09-15 to 2025-09-15")
    print("=" * 80)

    symbols = ['OPEN', 'TLT', 'TQQQ', 'TSLA', 'META', 'NVDA']
    start_date = '2020-09-15'
    end_date = '2025-09-15'

    analyzer = QuantitativeAnalyzer(symbols, start_date, end_date)

    try:
        # 1) Fetch data
        if not analyzer.fetch_data():
            print("Data fetch failed, exiting")
            return

        # 2) Returns
        analyzer.calculate_returns()

        # 3) Correlation analysis
        analyzer.correlation_analysis()

        # 4) Sharpe ratio analysis
        analyzer.sharpe_ratio_analysis()

        # 5) Portfolio strategies
        analyzer.portfolio_strategies()

        # 6) Additional analysis
        analyzer.additional_analysis()

        # 7) Dashboard
        analyzer.comprehensive_dashboard()

        print("\n" + "=" * 80)
        print("Quantitative analysis completed. All figures saved to the figures/ directory")
        print("=" * 80)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

