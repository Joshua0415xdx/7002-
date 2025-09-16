import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import os
from datetime import datetime

# Set matplotlib style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class QuantitativeAnalyzer:

    def __init__(self, symbols, start_date, end_date, benchmark='SPY'):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.data = None
        self.returns = None
        self.benchmark_data = None
        self.benchmark_returns = None

    def fetch_data(self):
        print('Fetching stock data...')

        try:
            raw_data = yf.download(self.symbols, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)

            if raw_data.columns.nlevels > 1:
                self.data = raw_data['Close'].copy()
            else:
                self.data = raw_data.copy()

            print(f'Successfully fetched data for {len(self.data.columns)} stocks')

            benchmark_raw = yf.download(self.benchmark, start=self.start_date, end=self.end_date, progress=False)

            if benchmark_raw.columns.nlevels > 1:
                self.benchmark_data = benchmark_raw[('Close', self.benchmark)]
            else:
                self.benchmark_data = benchmark_raw['Close']

            print(f"Successfully fetched benchmark data for {self.benchmark}")

            print("\nData integrity check:")
            for symbol in self.data.columns:
                missing_count = self.data[symbol].isnull().sum()
                total_points = len(self.data[symbol])
                valid_points = total_points - missing_count

                if missing_count > 0:
                    print(f"{symbol}: Valid data {valid_points}/{total_points} points")
                else:
                    print(f"{symbol}: Complete data ({total_points} points)")

            valid_data_start = self.data.dropna().index[0] if len(self.data.dropna()) > 0 else self.data.index[0]

            self.data = self.data.loc[valid_data_start:].fillna(method='ffill').dropna(how='all')
            self.benchmark_data = self.benchmark_data.loc[valid_data_start:].fillna(method='ffill').dropna()

            # Align benchmark data and stock data time indices
            common_dates = self.data.index.intersection(self.benchmark_data.index)
            self.data = self.data.loc[common_dates]
            self.benchmark_data = self.benchmark_data.loc[common_dates]

            print(f"\nFinal data time range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
            print(f"Total {len(self.data)} trading days")

            # Final data quality check
            final_missing = self.data.isnull().sum()
            if final_missing.sum() > 0:
                print("\nFinal data missing situation:")
                for symbol, missing in final_missing.items():
                    if missing > 0:
                        print(f"{symbol}: {missing} missing values")
            else:
                print("\nAll data has been completely aligned")

        except Exception as e:
            print(f"Data fetching failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    def calculate_returns(self):
        """Calculate daily returns"""
        print("\nCalculating returns...")

        # Calculate stock daily returns
        self.returns = self.data.pct_change().dropna()

        # Calculate benchmark returns
        self.benchmark_returns = self.benchmark_data.pct_change().dropna()

        print("Return statistics summary:")
        print(self.returns.describe())

    def correlation_analysis(self):
        """Correlation analysis - Calculate Pearson correlation coefficient"""
        print("\nPerforming correlation analysis...")

        # Calculate correlation coefficient matrix
        correlation_matrix = self.returns.corr()

        # Create heatmap with improved styling
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Enhanced heatmap with better colors and formatting
        sns.heatmap(correlation_matrix,
                    mask=mask,
                    annot=True,
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    fmt='.3f',
                    cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                    linewidths=0.5,
                    annot_kws={'size': 10})
        
        plt.title('Stock Return Correlation Heatmap\n(Pearson Correlation Coefficient)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Stocks', fontsize=12)
        plt.ylabel('Stocks', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save to current directory instead of hardcoded path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'correlation_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

        return correlation_matrix

    def sharpe_ratio_analysis(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio - Risk-adjusted returns"""
        print("\nCalculating Sharpe ratios...")

        # Calculate annualized returns
        annual_returns = self.returns.mean() * 252

        # Calculate annualized volatility
        annual_volatility = self.returns.std() * np.sqrt(252)

        # Calculate Sharpe ratios
        sharpe_ratios = (annual_returns - risk_free_rate) / annual_volatility

        # Create enhanced charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Sharpe ratio bar chart with improved styling
        colors = ['#2E8B57' if x > 0 else '#DC143C' for x in sharpe_ratios]
        bars = sharpe_ratios.plot(kind='bar', ax=ax1, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_title('Sharpe Ratio (Risk-Adjusted Returns)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Sharpe Ratio', fontsize=12)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Stocks', fontsize=12)
        
        # Add value labels on bars
        for bar in ax1.patches:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=9, fontweight='bold')

        # Risk-return scatter plot with enhanced styling
        scatter = ax2.scatter(annual_volatility, annual_returns, s=120, alpha=0.8, 
                            c=range(len(self.symbols)), cmap='viridis', edgecolors='black', linewidth=1)
        
        # Add stock labels with better positioning
        for i, symbol in enumerate(self.symbols):
            ax2.annotate(symbol, (annual_volatility.iloc[i], annual_returns.iloc[i]),
                        xytext=(8, 8), textcoords='offset points', 
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Annualized Volatility', fontsize=12)
        ax2.set_ylabel('Annualized Return', fontsize=12)
        ax2.set_title('Risk-Return Scatter Plot', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3)
        
        # Add efficient frontier reference line
        x_line = np.linspace(annual_volatility.min(), annual_volatility.max(), 100)
        y_line = risk_free_rate + (annual_returns.max() - risk_free_rate) * (x_line / annual_volatility.max())
        ax2.plot(x_line, y_line, '--', color='gray', alpha=0.5, label='Reference Line')
        ax2.legend()

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'sharpe_ratio_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Sharpe Ratio Rankings:")
        print(sharpe_ratios.sort_values(ascending=False))

        return sharpe_ratios, annual_returns, annual_volatility

    def portfolio_strategies(self):
        """Portfolio strategy simulation"""
        print("\nPerforming portfolio strategy analysis...")

        # 1. Equal weight portfolio
        equal_weight_returns = self.returns.mean(axis=1)

        # 2. Minimum variance portfolio (simplified version using inverse correlation as weights)
        correlation_matrix = self.returns.corr()
        # Calculate average correlation for each asset, lower correlation gets higher weight
        avg_correlations = correlation_matrix.mean(axis=1)
        min_var_weights = (1 / avg_correlations) / (1 / avg_correlations).sum()
        min_var_returns = (self.returns * min_var_weights).sum(axis=1)

        # 3. Momentum strategy (based on past 20-day returns)
        momentum_scores = self.returns.rolling(window=20).mean()
        momentum_weights = momentum_scores.div(momentum_scores.sum(axis=1), axis=0)
        momentum_returns = (self.returns * momentum_weights.shift(1)).sum(axis=1)

        # 4. Benchmark returns
        benchmark_returns = self.benchmark_returns

        # Calculate cumulative returns
        strategies = {
            'Equal Weight Portfolio': equal_weight_returns,
            'Minimum Variance Portfolio': min_var_returns,
            'Momentum Strategy': momentum_returns.dropna(),
            f'{self.benchmark} Benchmark': benchmark_returns
        }

        cumulative_returns = {}
        for name, returns in strategies.items():
            cumulative_returns[name] = (1 + returns).cumprod()

        # Create enhanced strategy comparison charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # Color palette for strategies
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # Cumulative return curves with enhanced styling
        for i, (name, cum_ret) in enumerate(cumulative_returns.items()):
            ax1.plot(cum_ret.index, cum_ret.values, linewidth=2.5, 
                    label=name, alpha=0.9, color=colors[i % len(colors)])
        ax1.set_title('Portfolio Strategy Cumulative Return Comparison', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Annualized return comparison
        annual_rets = {name: ret.mean() * 252 for name, ret in strategies.items()}
        annual_rets_series = pd.Series(annual_rets)
        bars1 = annual_rets_series.plot(kind='bar', ax=ax2, color=colors, alpha=0.8, 
                                       edgecolor='black', linewidth=0.5)
        ax2.set_title('Annualized Return by Strategy', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('Annualized Return', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in ax2.patches:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        # Volatility comparison
        volatilities = {name: ret.std() * np.sqrt(252) for name, ret in strategies.items()}
        vol_series = pd.Series(volatilities)
        bars2 = vol_series.plot(kind='bar', ax=ax3, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'], 
                               alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.set_title('Annualized Volatility by Strategy', fontsize=14, fontweight='bold', pad=15)
        ax3.set_ylabel('Annualized Volatility', fontsize=12)
        ax3.tick_params(axis='x', rotation=45, labelsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in ax3.patches:
            height = bar.get_height()
            ax3.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        # Sharpe ratio comparison
        sharpe_ratios = {name: (annual_rets[name] - 0.02) / volatilities[name]
                        for name in annual_rets.keys()}
        sharpe_series = pd.Series(sharpe_ratios)
        colors_sharpe = ['#2E8B57' if x > 0 else '#DC143C' for x in sharpe_series]
        bars3 = sharpe_series.plot(kind='bar', ax=ax4, color=colors_sharpe, alpha=0.8,
                                  edgecolor='black', linewidth=0.5)
        ax4.set_title('Sharpe Ratio by Strategy', fontsize=14, fontweight='bold', pad=15)
        ax4.set_ylabel('Sharpe Ratio', fontsize=12)
        ax4.tick_params(axis='x', rotation=45, labelsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in ax4.patches:
            height = bar.get_height()
            ax4.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=9, fontweight='bold')

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'portfolio_strategies_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print strategy statistics
        print("\nPortfolio Strategy Statistics:")
        strategy_stats = pd.DataFrame({
            'Annualized Return': annual_rets_series,
            'Annualized Volatility': vol_series,
            'Sharpe Ratio': sharpe_series
        })
        print(strategy_stats)

        return strategy_stats, min_var_weights

    def additional_analysis(self):
        """Additional quantitative analysis methods"""
        print("\nPerforming additional quantitative analysis...")

        # 1. VaR calculation (Value at Risk)
        confidence_level = 0.05
        var_5 = self.returns.quantile(confidence_level)

        # 2. Maximum drawdown calculation
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 3. 60-day rolling correlation (with benchmark)
        rolling_corr = {}
        for symbol in self.symbols:
            if symbol in self.returns.columns:
                rolling_corr[symbol] = self.returns[symbol].rolling(window=60).corr(self.benchmark_returns)
        rolling_corr_df = pd.DataFrame(rolling_corr)

        # Create enhanced charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        # VaR bar chart with enhanced styling
        bars1 = var_5.plot(kind='bar', ax=ax1, color='#DC143C', alpha=0.8, 
                          edgecolor='black', linewidth=0.5)
        ax1.set_title('5% Value at Risk (VaR)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('VaR', fontsize=12)
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Stocks', fontsize=12)
        
        # Add value labels
        for bar in ax1.patches:
            height = bar.get_height()
            ax1.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -15),
                        textcoords="offset points",
                        ha='center', va='top',
                        fontsize=9, fontweight='bold')

        # Maximum drawdown bar chart
        bars2 = max_drawdown.plot(kind='bar', ax=ax2, color='#8B0000', alpha=0.8,
                                 edgecolor='black', linewidth=0.5)
        ax2.set_title('Maximum Drawdown', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('Maximum Drawdown', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Stocks', fontsize=12)
        
        # Add value labels
        for bar in ax2.patches:
            height = bar.get_height()
            ax2.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -15),
                        textcoords="offset points",
                        ha='center', va='top',
                        fontsize=9, fontweight='bold')

        # Drawdown time series with enhanced styling
        colors_dd = plt.cm.Set1(np.linspace(0, 1, len(self.symbols)))
        for i, symbol in enumerate(self.symbols):
            if symbol in drawdown.columns:
                ax3.fill_between(drawdown.index, drawdown[symbol], 0, 
                               alpha=0.6, color=colors_dd[i], label=symbol)
                ax3.plot(drawdown.index, drawdown[symbol], 
                        linewidth=1.5, color=colors_dd[i])
        ax3.set_title('Drawdown Time Series', fontsize=14, fontweight='bold', pad=15)
        ax3.set_ylabel('Drawdown Ratio', fontsize=12)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # 60-day rolling correlation with enhanced styling
        colors_corr = plt.cm.tab10(np.linspace(0, 1, len(rolling_corr_df.columns)))
        for i, column in enumerate(rolling_corr_df.columns):
            ax4.plot(rolling_corr_df.index, rolling_corr_df[column], 
                    linewidth=2, alpha=0.8, color=colors_corr[i], label=column)
        ax4.set_title(f'60-Day Rolling Correlation (vs {self.benchmark})', 
                     fontsize=14, fontweight='bold', pad=15)
        ax4.set_ylabel('Correlation Coefficient', fontsize=12)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'additional_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("5% VaR:")
        print(var_5.sort_values())
        print("\nMaximum Drawdown:")
        print(max_drawdown.sort_values())

        return var_5, max_drawdown, rolling_corr_df

    def comprehensive_dashboard(self):
        """Create comprehensive analysis dashboard"""
        print("\nCreating comprehensive analysis dashboard...")

        # Calculate key metrics
        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (annual_returns - 0.02) / annual_volatility

        # Create comprehensive dashboard with improved layout
        fig = plt.figure(figsize=(24, 18))
        
        # Use GridSpec for better layout control
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Normalized price trend
        ax1 = fig.add_subplot(gs[0, 0])
        normalized_prices = self.data / self.data.iloc[0]
        colors_price = plt.cm.tab10(np.linspace(0, 1, len(self.symbols)))
        for i, column in enumerate(normalized_prices.columns):
            ax1.plot(normalized_prices.index, normalized_prices[column], 
                    linewidth=2, alpha=0.8, color=colors_price[i], label=column)
        ax1.set_title('Normalized Price Trend\n(Base Period = 1)', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('Normalized Price', fontsize=10)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)

        # 2. Return distribution box plot
        ax2 = fig.add_subplot(gs[0, 1])
        box_plot = self.returns.plot(kind='box', ax=ax2, patch_artist=True)
        ax2.set_title('Return Distribution Box Plot', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Daily Return', fontsize=10)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Color the box plots
        colors_box = plt.cm.Set3(np.linspace(0, 1, len(self.symbols)))
        for patch, color in zip(box_plot.patches, colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 3. Correlation heatmap (simplified)
        ax3 = fig.add_subplot(gs[0, 2])
        correlation_matrix = self.returns.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax3,
                   linewidths=0.5, annot_kws={'size': 8})
        ax3.set_title('Return Correlation Matrix', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
        ax3.tick_params(axis='y', rotation=0, labelsize=8)

        # 4. Annualized returns
        ax4 = fig.add_subplot(gs[1, 0])
        bars4 = annual_returns.plot(kind='bar', ax=ax4, color='steelblue', alpha=0.8,
                                   edgecolor='black', linewidth=0.5)
        ax4.set_title('Annualized Return', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Annualized Return', fontsize=10)
        ax4.tick_params(axis='x', rotation=45, labelsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in ax4.patches:
            height = bar.get_height()
            ax4.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

        # 5. Annualized volatility
        ax5 = fig.add_subplot(gs[1, 1])
        bars5 = annual_volatility.plot(kind='bar', ax=ax5, color='orange', alpha=0.8,
                                      edgecolor='black', linewidth=0.5)
        ax5.set_title('Annualized Volatility', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Annualized Volatility', fontsize=10)
        ax5.tick_params(axis='x', rotation=45, labelsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in ax5.patches:
            height = bar.get_height()
            ax5.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

        # 6. Sharpe ratio
        ax6 = fig.add_subplot(gs[1, 2])
        colors_sharpe = ['#2E8B57' if x > 0 else '#DC143C' for x in sharpe_ratios]
        bars6 = sharpe_ratios.plot(kind='bar', ax=ax6, color=colors_sharpe, alpha=0.8,
                                  edgecolor='black', linewidth=0.5)
        ax6.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Sharpe Ratio', fontsize=10)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax6.tick_params(axis='x', rotation=45, labelsize=8)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in ax6.patches:
            height = bar.get_height()
            ax6.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=8, fontweight='bold')

        # 7. Risk-Return scatter
        ax7 = fig.add_subplot(gs[2, 0])
        scatter = ax7.scatter(annual_volatility, annual_returns, s=100, alpha=0.8, 
                            c=range(len(self.symbols)), cmap='viridis', 
                            edgecolors='black', linewidth=1)
        for i, symbol in enumerate(self.symbols):
            ax7.annotate(symbol, (annual_volatility.iloc[i], annual_returns.iloc[i]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax7.set_xlabel('Annualized Volatility', fontsize=10)
        ax7.set_ylabel('Annualized Return', fontsize=10)
        ax7.set_title('Risk-Return Analysis', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # 8. Rolling volatility
        ax8 = fig.add_subplot(gs[2, 1])
        rolling_vol = self.returns.rolling(window=30).std() * np.sqrt(252)
        colors_vol = plt.cm.tab10(np.linspace(0, 1, len(self.symbols)))
        for i, column in enumerate(rolling_vol.columns):
            ax8.plot(rolling_vol.index, rolling_vol[column], 
                    linewidth=1.5, alpha=0.8, color=colors_vol[i], label=column)
        ax8.set_title('30-Day Rolling Volatility', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Annualized Volatility', fontsize=10)
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax8.grid(True, alpha=0.3)
        ax8.tick_params(axis='x', rotation=45, labelsize=8)

        # 9. Cumulative return comparison
        ax9 = fig.add_subplot(gs[2, 2])
        cumulative_returns = (1 + self.returns).cumprod()
        colors_cum = plt.cm.tab10(np.linspace(0, 1, len(self.symbols)))
        for i, column in enumerate(cumulative_returns.columns):
            ax9.plot(cumulative_returns.index, cumulative_returns[column], 
                    linewidth=2, alpha=0.8, color=colors_cum[i], label=column)
        
        benchmark_cum = (1 + self.benchmark_returns).cumprod()
        ax9.plot(benchmark_cum.index, benchmark_cum.values,
                linewidth=2.5, alpha=0.9, label=f'{self.benchmark} Benchmark', 
                color='black', linestyle='--')
        ax9.set_title('Cumulative Return Comparison', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Cumulative Return', fontsize=10)
        ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax9.grid(True, alpha=0.3)
        ax9.tick_params(axis='x', rotation=45, labelsize=8)

        plt.suptitle('Comprehensive Stock Quantitative Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'comprehensive_dashboard_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function - Execute complete quantitative analysis workflow"""
    print("=" * 80)
    print("STOCK QUANTITATIVE ANALYSIS SYSTEM")
    print("Analyzing stocks: OPEN, TLT, TQQQ, TSLA, META, NVDA")
    print("Time range: September 15, 2020 - September 15, 2025")
    print("=" * 80)

    # Define stock symbols and time range
    symbols = ['OPEN', 'TLT', 'TQQQ', 'TSLA', 'META', 'NVDA']
    start_date = '2020-09-15'
    end_date = '2025-09-15'

    # Create analyzer instance
    analyzer = QuantitativeAnalyzer(symbols, start_date, end_date)

    # Execute analysis workflow
    try:
        # 1. Fetch data
        if not analyzer.fetch_data():
            print("Data fetching failed, program terminated")
            return

        # 2. Calculate returns
        analyzer.calculate_returns()

        # 3. Correlation analysis
        correlation_matrix = analyzer.correlation_analysis()

        # 4. Sharpe ratio analysis
        sharpe_ratios, annual_returns, annual_volatility = analyzer.sharpe_ratio_analysis()

        # 5. Portfolio strategies
        strategy_stats, min_var_weights = analyzer.portfolio_strategies()

        # 6. Additional analysis
        var_5, max_drawdown, rolling_corr_df = analyzer.additional_analysis()

        # 7. Comprehensive dashboard
        analyzer.comprehensive_dashboard()

        print("\n" + "=" * 80)
        print("QUANTITATIVE ANALYSIS COMPLETED!")
        print("All charts have been saved to the current working directory")
        print("=" * 80)

    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()