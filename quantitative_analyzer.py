import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import os
from datetime import datetime

# Set matplotlib style for better visual appearance
plt.style.use('seaborn-v0_8')
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
        
        # Create output directory for charts
        self.output_dir = 'analysis_charts'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fetch_data(self):
        print('Fetching stock data...')

        try:
            raw_data = yf.download(self.symbols, start=self.start_date, end=self.end_date, 
                                 progress=False, auto_adjust=True)

            if raw_data.columns.nlevels > 1:
                self.data = raw_data['Close'].copy()
            else:
                self.data = raw_data.copy()

            print(f'Successfully fetched data for {len(self.data.columns)} stocks')

            benchmark_raw = yf.download(self.benchmark, start=self.start_date, end=self.end_date, 
                                      progress=False)

            if benchmark_raw.columns.nlevels > 1:
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
                    print(f"{symbol}: {valid_points}/{total_points} valid data points")
                else:
                    print(f"{symbol}: Complete data ({total_points} data points)")

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
                print("\nFinal data missing values:")
                for symbol, missing in final_missing.items():
                    if missing > 0:
                        print(f"{symbol}: {missing} missing values")
            else:
                print("\nAll data has been properly aligned")

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

        print("Returns statistics summary:")
        print(self.returns.describe())

    def correlation_analysis(self):
        """Correlation analysis - Calculate Pearson correlation coefficient"""
        print("\nPerforming correlation analysis...")

        # Calculate correlation matrix
        correlation_matrix = self.returns.corr()

        # Create heatmap with improved styling
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Create custom colormap
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        sns.heatmap(correlation_matrix,
                    mask=mask,
                    annot=True,
                    cmap=cmap,
                    center=0,
                    square=True,
                    fmt='.3f',
                    cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                    linewidths=0.5,
                    annot_kws={'size': 10, 'weight': 'bold'})
        
        plt.title('Stock Returns Correlation Heatmap\n(Pearson Correlation Coefficient)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Stocks', fontsize=12, fontweight='bold')
        plt.ylabel('Stocks', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save with proper filename
        save_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        return correlation_matrix

    def sharpe_ratio_analysis(self, risk_free_rate=0.02):
        """Calculate Sharpe Ratio - Risk-adjusted returns"""
        print("\nCalculating Sharpe Ratio...")

        # Calculate annualized returns
        annual_returns = self.returns.mean() * 252

        # Calculate annualized volatility
        annual_volatility = self.returns.std() * np.sqrt(252)

        # Calculate Sharpe ratios
        sharpe_ratios = (annual_returns - risk_free_rate) / annual_volatility

        # Create improved visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Sharpe Ratio bar chart with improved styling
        colors = ['#2E8B57' if x > 0 else '#DC143C' for x in sharpe_ratios]
        bars = sharpe_ratios.plot(kind='bar', ax=ax1, color=colors, alpha=0.8, 
                                 edgecolor='black', linewidth=1.2)
        ax1.set_title('Sharpe Ratio\n(Risk-Adjusted Returns)', fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Stocks', fontsize=12, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(sharpe_ratios.items()):
            ax1.text(i, val + (0.05 if val >= 0 else -0.05), f'{val:.3f}', 
                    ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=9)

        # Risk-Return scatter plot with improved styling
        scatter = ax2.scatter(annual_volatility, annual_returns, s=150, alpha=0.8, 
                             c='steelblue', edgecolors='black', linewidth=1.5)
        
        for i, symbol in enumerate(self.symbols):
            ax2.annotate(symbol, (annual_volatility.iloc[i], annual_returns.iloc[i]),
                        xytext=(8, 8), textcoords='offset points', fontsize=11, 
                        fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax2.set_xlabel('Annualized Volatility', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Annualized Returns', fontsize=12, fontweight='bold')
        ax2.set_title('Risk-Return Scatter Plot', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(annual_volatility, annual_returns, 1)
        p = np.poly1d(z)
        ax2.plot(annual_volatility, p(annual_volatility), "r--", alpha=0.8, linewidth=2)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'sharpe_ratio_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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
            'Equal Weight': equal_weight_returns,
            'Min Variance': min_var_returns,
            'Momentum': momentum_returns.dropna(),
            f'{self.benchmark} Benchmark': benchmark_returns
        }

        cumulative_returns = {}
        for name, returns in strategies.items():
            cumulative_returns[name] = (1 + returns).cumprod()

        # Create improved strategy comparison charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))

        # Cumulative returns curve with improved styling
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (name, cum_ret) in enumerate(cumulative_returns.items()):
            ax1.plot(cum_ret.index, cum_ret.values, linewidth=3, label=name, 
                    alpha=0.9, color=colors[i % len(colors)])
        
        ax1.set_title('Portfolio Strategy Cumulative Returns Comparison', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('Cumulative Returns', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Annualized returns comparison with improved styling
        annual_rets = {name: ret.mean() * 252 for name, ret in strategies.items()}
        annual_rets_series = pd.Series(annual_rets)
        bars = annual_rets_series.plot(kind='bar', ax=ax2, color='skyblue', alpha=0.8,
                                      edgecolor='black', linewidth=1.2)
        ax2.set_title('Annualized Returns by Strategy', fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('Annualized Returns', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Strategy', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(annual_rets_series.items()):
            ax2.text(i, val + (0.01 if val >= 0 else -0.01), f'{val:.3f}', 
                    ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=9)

        # Volatility comparison with improved styling
        volatilities = {name: ret.std() * np.sqrt(252) for name, ret in strategies.items()}
        vol_series = pd.Series(volatilities)
        bars = vol_series.plot(kind='bar', ax=ax3, color='lightcoral', alpha=0.8,
                              edgecolor='black', linewidth=1.2)
        ax3.set_title('Annualized Volatility by Strategy', fontsize=14, fontweight='bold', pad=20)
        ax3.set_ylabel('Annualized Volatility', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Strategy', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45, labelsize=10)
        ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(vol_series.items()):
            ax3.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)

        # Sharpe ratio comparison with improved styling
        sharpe_ratios = {name: (annual_rets[name] - 0.02) / volatilities[name]
                         for name in annual_rets.keys()}
        sharpe_series = pd.Series(sharpe_ratios)
        colors = ['#2E8B57' if x > 0 else '#DC143C' for x in sharpe_series]
        bars = sharpe_series.plot(kind='bar', ax=ax4, color=colors, alpha=0.8,
                                 edgecolor='black', linewidth=1.2)
        ax4.set_title('Sharpe Ratio by Strategy', fontsize=14, fontweight='bold', pad=20)
        ax4.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Strategy', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45, labelsize=10)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(sharpe_series.items()):
            ax4.text(i, val + (0.05 if val >= 0 else -0.05), f'{val:.3f}', 
                    ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'portfolio_strategies.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        # Print strategy statistics
        print("\nPortfolio Strategy Statistics:")
        strategy_stats = pd.DataFrame({
            'Annualized Returns': annual_rets_series,
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

        # 3. 60-day rolling correlation (with SPY)
        rolling_corr = {}
        for symbol in self.symbols:
            if symbol in self.returns.columns:
                rolling_corr[symbol] = self.returns[symbol].rolling(window=60).corr(self.benchmark_returns)
        rolling_corr_df = pd.DataFrame(rolling_corr)

        # Create improved charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))

        # VaR bar chart with improved styling
        bars = var_5.plot(kind='bar', ax=ax1, color='red', alpha=0.8, 
                         edgecolor='black', linewidth=1.2)
        ax1.set_title('5% Value at Risk (VaR)', fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('VaR', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Stocks', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(var_5.items()):
            ax1.text(i, val - 0.01, f'{val:.3f}', ha='center', va='top', 
                    fontweight='bold', fontsize=9)

        # Maximum drawdown bar chart with improved styling
        bars = max_drawdown.plot(kind='bar', ax=ax2, color='darkred', alpha=0.8,
                                edgecolor='black', linewidth=1.2)
        ax2.set_title('Maximum Drawdown', fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('Maximum Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Stocks', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(max_drawdown.items()):
            ax2.text(i, val - 0.01, f'{val:.3f}', ha='center', va='top', 
                    fontweight='bold', fontsize=9)

        # Drawdown time series with improved styling
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.symbols)))
        for i, symbol in enumerate(self.symbols):
            if symbol in drawdown.columns:
                ax3.plot(drawdown.index, drawdown[symbol], linewidth=2.5, 
                        alpha=0.9, label=symbol, color=colors[i])
        
        ax3.set_title('Drawdown Time Series', fontsize=14, fontweight='bold', pad=20)
        ax3.set_ylabel('Drawdown Ratio', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

        # 60-day rolling correlation with improved styling
        colors = plt.cm.tab10(np.linspace(0, 1, len(rolling_corr_df.columns)))
        for i, col in enumerate(rolling_corr_df.columns):
            ax4.plot(rolling_corr_df.index, rolling_corr_df[col], 
                    linewidth=2.5, alpha=0.9, label=col, color=colors[i])
        
        ax4.set_title(f'60-Day Rolling Correlation\n(Relative to {self.benchmark})', 
                     fontsize=14, fontweight='bold', pad=20)
        ax4.set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'additional_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
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

        # 1. Price trend chart with improved styling
        ax1 = plt.subplot(3, 3, 1)
        normalized_prices = self.data / self.data.iloc[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(normalized_prices.columns)))
        for i, col in enumerate(normalized_prices.columns):
            ax1.plot(normalized_prices.index, normalized_prices[col], 
                    linewidth=2.5, alpha=0.9, label=col, color=colors[i])
        
        ax1.set_title('Normalized Price Trends\n(Base Period = 1)', fontsize=12, fontweight='bold', pad=15)
        ax1.set_ylabel('Normalized Prices', fontsize=10, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # 2. Returns distribution with improved styling
        ax2 = plt.subplot(3, 3, 2)
        box_plot = self.returns.plot(kind='box', ax=ax2, patch_artist=True)
        ax2.set_title('Returns Distribution Box Plot', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylabel('Daily Returns', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Stocks', fontsize=10, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.returns.columns)))
        for patch, color in zip(box_plot.artists, colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 3. Correlation heatmap (simplified version) with improved styling
        ax3 = plt.subplot(3, 3, 3)
        correlation_matrix = self.returns.corr()
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax3,
                    linewidths=0.5, annot_kws={'size': 8, 'weight': 'bold'})
        ax3.set_title('Returns Correlation', fontsize=12, fontweight='bold', pad=15)

        # 4. Annualized returns with improved styling
        ax4 = plt.subplot(3, 3, 4)
        bars = annual_returns.plot(kind='bar', ax=ax4, color='steelblue', alpha=0.8,
                                  edgecolor='black', linewidth=1.2)
        ax4.set_title('Annualized Returns', fontsize=12, fontweight='bold', pad=15)
        ax4.set_ylabel('Annualized Returns', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Stocks', fontsize=10, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45, labelsize=8)
        ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add value labels
        for i, (idx, val) in enumerate(annual_returns.items()):
            ax4.text(i, val + (0.01 if val >= 0 else -0.01), f'{val:.3f}', 
                    ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=8)

        # 5. Annualized volatility with improved styling
        ax5 = plt.subplot(3, 3, 5)
        bars = annual_volatility.plot(kind='bar', ax=ax5, color='lightcoral', alpha=0.8,
                                     edgecolor='black', linewidth=1.2)
        ax5.set_title('Annualized Volatility', fontsize=12, fontweight='bold', pad=15)
        ax5.set_ylabel('Annualized Volatility', fontsize=10, fontweight='bold')
        ax5.set_xlabel('Stocks', fontsize=10, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45, labelsize=8)
        ax5.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add value labels
        for i, (idx, val) in enumerate(annual_volatility.items()):
            ax5.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=8)

        # 6. Sharpe ratio with improved styling
        ax6 = plt.subplot(3, 3, 6)
        colors = ['#2E8B57' if x > 0 else '#DC143C' for x in sharpe_ratios]
        bars = sharpe_ratios.plot(kind='bar', ax=ax6, color=colors, alpha=0.8,
                                 edgecolor='black', linewidth=1.2)
        ax6.set_title('Sharpe Ratio', fontsize=12, fontweight='bold', pad=15)
        ax6.set_ylabel('Sharpe Ratio', fontsize=10, fontweight='bold')
        ax6.set_xlabel('Stocks', fontsize=10, fontweight='bold')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax6.tick_params(axis='x', rotation=45, labelsize=8)
        ax6.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add value labels
        for i, (idx, val) in enumerate(sharpe_ratios.items()):
            ax6.text(i, val + (0.05 if val >= 0 else -0.05), f'{val:.3f}', 
                    ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=8)

        # 7. Risk-return scatter plot
        ax7 = plt.subplot(3, 3, 7)
        scatter = ax7.scatter(annual_volatility, annual_returns, s=120, alpha=0.8, 
                             c='steelblue', edgecolors='black', linewidth=1.5)
        
        for i, symbol in enumerate(self.symbols):
            ax7.annotate(symbol, (annual_volatility.iloc[i], annual_returns.iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, 
                        fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax7.set_xlabel('Annualized Volatility', fontsize=10, fontweight='bold')
        ax7.set_ylabel('Annualized Returns', fontsize=10, fontweight='bold')
        ax7.set_title('Risk-Return Scatter Plot', fontsize=12, fontweight='bold', pad=15)
        ax7.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # 8. Rolling volatility (30-day)
        ax8 = plt.subplot(3, 3, 8)
        rolling_vol = self.returns.rolling(window=30).std() * np.sqrt(252)
        colors = plt.cm.tab10(np.linspace(0, 1, len(rolling_vol.columns)))
        for i, col in enumerate(rolling_vol.columns):
            ax8.plot(rolling_vol.index, rolling_vol[col], linewidth=2, 
                    alpha=0.9, label=col, color=colors[i])
        
        ax8.set_title('30-Day Rolling Volatility', fontsize=12, fontweight='bold', pad=15)
        ax8.set_ylabel('Rolling Volatility', fontsize=10, fontweight='bold')
        ax8.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax8.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # 9. Cumulative returns comparison with improved styling
        ax9 = plt.subplot(3, 3, 9)
        cumulative_returns = (1 + self.returns).cumprod()
        colors = plt.cm.tab10(np.linspace(0, 1, len(cumulative_returns.columns)))
        for i, col in enumerate(cumulative_returns.columns):
            ax9.plot(cumulative_returns.index, cumulative_returns[col], 
                    linewidth=2.5, alpha=0.9, label=col, color=colors[i])
        
        benchmark_cum = (1 + self.benchmark_returns).cumprod()
        ax9.plot(benchmark_cum.index, benchmark_cum.values,
                linewidth=3, alpha=0.9, label=f'{self.benchmark} Benchmark', 
                color='black', linestyle='--')
        
        ax9.set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold', pad=15)
        ax9.set_ylabel('Cumulative Returns', fontsize=10, fontweight='bold')
        ax9.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax9.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        plt.suptitle('Stock Quantitative Analysis Comprehensive Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'comprehensive_dashboard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()


def main():
    """Main function - Execute complete quantitative analysis workflow"""
    print("=" * 80)
    print("Stock Quantitative Analysis System")
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
            print("Data fetching failed, program exiting")
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
        print("Quantitative analysis completed! All charts saved to 'analysis_charts' directory")
        print("=" * 80)

    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()