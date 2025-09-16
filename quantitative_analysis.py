#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化股票分析程序
分析股票：OPEN, SERV, GLD, TLT, TQQQ, TSLA, META, NVDA
时间范围：2020年9月15日 - 2025年9月15日
作者：量化研究分析师
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class QuantitativeAnalyzer:
    """量化分析器类 - 用于进行股票量化分析"""
    
    def __init__(self, symbols, start_date, end_date, benchmark='SPY'):
        """
        初始化分析器
        :param symbols: 股票代码列表
        :param start_date: 开始日期
        :param end_date: 结束日期  
        :param benchmark: 基准指数，默认为SPY
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.data = None
        self.returns = None
        self.benchmark_data = None
        self.benchmark_returns = None
        
    def fetch_data(self):
        """从Yahoo Finance获取股票数据"""
        print("正在获取股票数据...")
        
        try:
            # 获取多只股票数据
            raw_data = yf.download(self.symbols, start=self.start_date, end=self.end_date, progress=False)
            
            # 从多层索引中提取收盘价数据
            if raw_data.columns.nlevels > 1:
                # 提取Close价格数据
                self.data = raw_data['Close'].copy()
            else:
                # 单层索引情况
                self.data = raw_data.copy()
            
            print(f"成功获取 {len(self.data.columns)} 只股票的数据")
            
            # 获取基准数据
            benchmark_raw = yf.download(self.benchmark, start=self.start_date, end=self.end_date, progress=False)
            if benchmark_raw.columns.nlevels > 1:
                self.benchmark_data = benchmark_raw[('Close', self.benchmark)]
            else:
                self.benchmark_data = benchmark_raw['Close']
            print(f"成功获取基准指数 {self.benchmark} 的数据")
            
            # 检查数据完整性
            print("\n数据完整性检查：")
            for symbol in self.data.columns:
                missing_count = self.data[symbol].isnull().sum()
                total_points = len(self.data[symbol])
                valid_points = total_points - missing_count
                if missing_count > 0:
                    print(f"{symbol}: 有效数据 {valid_points}/{total_points} 个点")
                else:
                    print(f"{symbol}: 数据完整 ({total_points} 个点)")
            
            # 检查SERV数据的特殊情况（新上市股票）
            if 'SERV' in self.data.columns:
                serv_data = self.data['SERV'].dropna()
                if len(serv_data) > 0:
                    print(f"SERV数据时间范围: {serv_data.index[0].strftime('%Y-%m-%d')} 至 {serv_data.index[-1].strftime('%Y-%m-%d')}")
                    if len(serv_data) < len(self.data) * 0.7:  # 如果SERV数据少于70%
                        print(f"注意：SERV是新上市股票，数据点较少")
            
            # 对齐所有数据到相同的时间范围
            # 找到所有股票都有数据的时间范围
            valid_data_start = self.data.dropna().index[0] if len(self.data.dropna()) > 0 else self.data.index[0]
            
            # 填充缺失值（向前填充）并删除完全缺失的行
            self.data = self.data.loc[valid_data_start:].fillna(method='ffill').dropna(how='all')
            self.benchmark_data = self.benchmark_data.loc[valid_data_start:].fillna(method='ffill').dropna()
            
            # 对齐基准数据和股票数据的时间索引
            common_dates = self.data.index.intersection(self.benchmark_data.index)
            self.data = self.data.loc[common_dates]
            self.benchmark_data = self.benchmark_data.loc[common_dates]
            
            print(f"\n最终数据时间范围: {self.data.index[0].strftime('%Y-%m-%d')} 至 {self.data.index[-1].strftime('%Y-%m-%d')}")
            print(f"总共 {len(self.data)} 个交易日")
            
            # 最终数据质量检查
            final_missing = self.data.isnull().sum()
            if final_missing.sum() > 0:
                print("\n最终数据缺失情况：")
                for symbol, missing in final_missing.items():
                    if missing > 0:
                        print(f"{symbol}: {missing} 个缺失值")
            else:
                print("\n所有数据已完整对齐")
            
        except Exception as e:
            print(f"数据获取失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def calculate_returns(self):
        """计算日收益率"""
        print("\n计算收益率...")
        
        # 计算股票日收益率
        self.returns = self.data.pct_change().dropna()
        
        # 计算基准收益率
        self.benchmark_returns = self.benchmark_data.pct_change().dropna()
        
        print("收益率统计摘要：")
        print(self.returns.describe())
        
    def correlation_analysis(self):
        """相关性分析 - 计算皮尔逊相关系数"""
        print("\n进行相关性分析...")
        
        # 计算相关系数矩阵
        correlation_matrix = self.returns.corr()
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        plt.title('股票收益率相关性热力图 (皮尔逊相关系数)', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('/workspace/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def pca_analysis(self):
        """主成分分析 - 因子降维"""
        print("\n进行主成分分析...")
        
        # 标准化数据
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(self.returns)
        
        # 进行PCA
        pca = PCA()
        pca_result = pca.fit_transform(returns_scaled)
        
        # 计算累计方差解释比
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # 绘制PCA结果
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 方差解释比
        ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, 
                alpha=0.7, color='steelblue')
        ax1.plot(range(1, len(cumulative_variance) + 1), 
                cumulative_variance, 'ro-', linewidth=2)
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('方差解释比')
        ax1.set_title('主成分分析 - 方差解释比')
        ax1.grid(True, alpha=0.3)
        
        # 前两个主成分的散点图
        ax2.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, color='darkgreen')
        ax2.set_xlabel(f'第一主成分 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
        ax2.set_ylabel(f'第二主成分 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
        ax2.set_title('前两个主成分散点图')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印主成分载荷
        print("前3个主成分的载荷：")
        components_df = pd.DataFrame(
            pca.components_[:3].T,
            columns=['PC1', 'PC2', 'PC3'],
            index=self.symbols
        )
        print(components_df)
        
        return pca, components_df
    
    def volatility_analysis(self):
        """波动率分析"""
        print("\n进行波动率分析...")
        
        # 计算年化波动率
        annual_volatility = self.returns.std() * np.sqrt(252)
        
        # 计算30天滚动波动率
        rolling_vol = self.returns.rolling(window=30).std() * np.sqrt(252)
        
        # 绘制波动率图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 年化波动率柱状图
        annual_volatility.plot(kind='bar', ax=ax1, color='coral', alpha=0.8)
        ax1.set_title('各股票年化波动率', fontsize=14)
        ax1.set_ylabel('年化波动率')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 30天滚动波动率时间序列
        rolling_vol.plot(ax=ax2, linewidth=2, alpha=0.8)
        ax2.set_title('30天滚动波动率时间序列', fontsize=14)
        ax2.set_ylabel('年化波动率')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/volatility_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("年化波动率排名：")
        print(annual_volatility.sort_values(ascending=False))
        
        return annual_volatility, rolling_vol
    
    def sharpe_ratio_analysis(self, risk_free_rate=0.02):
        """计算夏普比率 - 风险调整后收益"""
        print("\n计算夏普比率...")
        
        # 计算年化收益率
        annual_returns = self.returns.mean() * 252
        
        # 计算年化波动率
        annual_volatility = self.returns.std() * np.sqrt(252)
        
        # 计算夏普比率
        sharpe_ratios = (annual_returns - risk_free_rate) / annual_volatility
        
        # 绘制夏普比率图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 夏普比率柱状图
        colors = ['green' if x > 0 else 'red' for x in sharpe_ratios]
        sharpe_ratios.plot(kind='bar', ax=ax1, color=colors, alpha=0.8)
        ax1.set_title('夏普比率 (风险调整后收益)', fontsize=14)
        ax1.set_ylabel('夏普比率')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 风险-收益散点图
        ax2.scatter(annual_volatility, annual_returns, s=100, alpha=0.7, c='steelblue')
        for i, symbol in enumerate(self.symbols):
            ax2.annotate(symbol, (annual_volatility.iloc[i], annual_returns.iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax2.set_xlabel('年化波动率')
        ax2.set_ylabel('年化收益率')
        ax2.set_title('风险-收益散点图', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/sharpe_ratio_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("夏普比率排名：")
        print(sharpe_ratios.sort_values(ascending=False))
        
        return sharpe_ratios, annual_returns, annual_volatility
    
    def beta_analysis(self):
        """计算Beta系数与市场基准的关系"""
        print(f"\n计算相对于{self.benchmark}的Beta系数...")
        
        # 确保数据对齐
        aligned_data = pd.concat([self.returns, self.benchmark_returns], axis=1, join='inner')
        stock_returns = aligned_data.iloc[:, :-1]
        market_returns = aligned_data.iloc[:, -1]
        
        # 计算Beta系数
        betas = {}
        alphas = {}
        r_squared = {}
        
        for symbol in self.symbols:
            if symbol in stock_returns.columns:
                # 线性回归计算Beta
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    market_returns, stock_returns[symbol]
                )
                betas[symbol] = slope
                alphas[symbol] = intercept * 252  # 年化Alpha
                r_squared[symbol] = r_value ** 2
        
        # 转换为Series
        beta_series = pd.Series(betas)
        alpha_series = pd.Series(alphas)
        r2_series = pd.Series(r_squared)
        
        # 绘制Beta分析图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Beta系数柱状图
        colors = ['green' if x < 1 else 'red' if x > 1 else 'blue' for x in beta_series]
        beta_series.plot(kind='bar', ax=ax1, color=colors, alpha=0.8)
        ax1.set_title('Beta系数 (相对于SPY)', fontsize=12)
        ax1.set_ylabel('Beta系数')
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='市场Beta=1')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Alpha柱状图
        alpha_colors = ['green' if x > 0 else 'red' for x in alpha_series]
        alpha_series.plot(kind='bar', ax=ax2, color=alpha_colors, alpha=0.8)
        ax2.set_title('年化Alpha (超额收益)', fontsize=12)
        ax2.set_ylabel('Alpha')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # R²柱状图
        r2_series.plot(kind='bar', ax=ax3, color='purple', alpha=0.8)
        ax3.set_title('R² (与市场相关程度)', fontsize=12)
        ax3.set_ylabel('R²')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Beta-Alpha散点图
        ax4.scatter(beta_series, alpha_series, s=100, alpha=0.7, c='steelblue')
        for symbol in beta_series.index:
            ax4.annotate(symbol, (beta_series[symbol], alpha_series[symbol]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax4.set_xlabel('Beta系数')
        ax4.set_ylabel('年化Alpha')
        ax4.set_title('Beta vs Alpha散点图', fontsize=12)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/beta_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Beta系数排名：")
        print(beta_series.sort_values(ascending=False))
        
        return beta_series, alpha_series, r2_series
    
    def portfolio_strategies(self):
        """投资组合策略模拟"""
        print("\n进行投资组合策略分析...")
        
        # 1. 等权重组合
        equal_weight_returns = self.returns.mean(axis=1)
        
        # 2. 最小方差组合 (简化版本，使用相关性的倒数作为权重)
        correlation_matrix = self.returns.corr()
        # 计算每个资产的平均相关性，相关性越低权重越高
        avg_correlations = correlation_matrix.mean(axis=1)
        min_var_weights = (1 / avg_correlations) / (1 / avg_correlations).sum()
        min_var_returns = (self.returns * min_var_weights).sum(axis=1)
        
        # 3. 动量策略 (基于过去20天收益率)
        momentum_scores = self.returns.rolling(window=20).mean()
        momentum_weights = momentum_scores.div(momentum_scores.sum(axis=1), axis=0)
        momentum_returns = (self.returns * momentum_weights.shift(1)).sum(axis=1)
        
        # 4. 基准收益
        benchmark_returns = self.benchmark_returns
        
        # 计算累计收益
        strategies = {
            '等权重组合': equal_weight_returns,
            '最小方差组合': min_var_returns,
            '动量策略': momentum_returns.dropna(),
            f'{self.benchmark}基准': benchmark_returns
        }
        
        cumulative_returns = {}
        for name, returns in strategies.items():
            cumulative_returns[name] = (1 + returns).cumprod()
        
        # 绘制策略比较图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 累计收益曲线
        for name, cum_ret in cumulative_returns.items():
            ax1.plot(cum_ret.index, cum_ret.values, linewidth=2, label=name, alpha=0.8)
        ax1.set_title('投资组合策略累计收益比较', fontsize=14)
        ax1.set_ylabel('累计收益')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 年化收益率比较
        annual_rets = {name: ret.mean() * 252 for name, ret in strategies.items()}
        annual_rets_series = pd.Series(annual_rets)
        annual_rets_series.plot(kind='bar', ax=ax2, color='skyblue', alpha=0.8)
        ax2.set_title('各策略年化收益率', fontsize=14)
        ax2.set_ylabel('年化收益率')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 波动率比较
        volatilities = {name: ret.std() * np.sqrt(252) for name, ret in strategies.items()}
        vol_series = pd.Series(volatilities)
        vol_series.plot(kind='bar', ax=ax3, color='lightcoral', alpha=0.8)
        ax3.set_title('各策略年化波动率', fontsize=14)
        ax3.set_ylabel('年化波动率')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 夏普比率比较
        sharpe_ratios = {name: (annual_rets[name] - 0.02) / volatilities[name] 
                        for name in annual_rets.keys()}
        sharpe_series = pd.Series(sharpe_ratios)
        colors = ['green' if x > 0 else 'red' for x in sharpe_series]
        sharpe_series.plot(kind='bar', ax=ax4, color=colors, alpha=0.8)
        ax4.set_title('各策略夏普比率', fontsize=14)
        ax4.set_ylabel('夏普比率')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/portfolio_strategies.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印策略统计
        print("\n投资组合策略统计：")
        strategy_stats = pd.DataFrame({
            '年化收益率': annual_rets_series,
            '年化波动率': vol_series,
            '夏普比率': sharpe_series
        })
        print(strategy_stats)
        
        return strategy_stats, min_var_weights
    
    def additional_analysis(self):
        """其他量化分析方法"""
        print("\n进行其他量化分析...")
        
        # 1. VaR计算 (Value at Risk)
        confidence_level = 0.05
        var_5 = self.returns.quantile(confidence_level)
        
        # 2. 最大回撤计算
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 3. 60天滚动相关性（与SPY）
        rolling_corr = {}
        for symbol in self.symbols:
            if symbol in self.returns.columns:
                rolling_corr[symbol] = self.returns[symbol].rolling(window=60).corr(self.benchmark_returns)
        rolling_corr_df = pd.DataFrame(rolling_corr)
        
        # 绘制图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # VaR柱状图
        var_5.plot(kind='bar', ax=ax1, color='red', alpha=0.8)
        ax1.set_title('5% VaR (Value at Risk)', fontsize=14)
        ax1.set_ylabel('VaR')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 最大回撤柱状图
        max_drawdown.plot(kind='bar', ax=ax2, color='darkred', alpha=0.8)
        ax2.set_title('最大回撤', fontsize=14)
        ax2.set_ylabel('最大回撤')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 回撤时间序列
        for symbol in self.symbols:
            if symbol in drawdown.columns:
                ax3.plot(drawdown.index, drawdown[symbol], linewidth=1.5, alpha=0.8, label=symbol)
        ax3.set_title('回撤时间序列', fontsize=14)
        ax3.set_ylabel('回撤比例')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 60天滚动相关性
        rolling_corr_df.plot(ax=ax4, linewidth=2, alpha=0.8)
        ax4.set_title(f'60天滚动相关性 (相对于{self.benchmark})', fontsize=14)
        ax4.set_ylabel('相关系数')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/additional_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("5% VaR:")
        print(var_5.sort_values())
        print("\n最大回撤:")
        print(max_drawdown.sort_values())
        
        return var_5, max_drawdown, rolling_corr_df
    
    def comprehensive_dashboard(self):
        """创建综合分析仪表板"""
        print("\n创建综合分析仪表板...")
        
        # 计算关键指标
        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratios = (annual_returns - 0.02) / annual_volatility
        
        # 创建综合仪表板
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 价格走势图
        ax1 = plt.subplot(3, 3, 1)
        normalized_prices = self.data / self.data.iloc[0]
        normalized_prices.plot(ax=ax1, linewidth=2, alpha=0.8)
        ax1.set_title('标准化价格走势 (基期=1)', fontsize=12)
        ax1.set_ylabel('标准化价格')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. 收益率分布
        ax2 = plt.subplot(3, 3, 2)
        self.returns.plot(kind='box', ax=ax2)
        ax2.set_title('收益率分布箱线图', fontsize=12)
        ax2.set_ylabel('日收益率')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. 相关性热力图（简化版）
        ax3 = plt.subplot(3, 3, 3)
        correlation_matrix = self.returns.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax3)
        ax3.set_title('收益率相关性', fontsize=12)
        
        # 4. 年化收益率
        ax4 = plt.subplot(3, 3, 4)
        annual_returns.plot(kind='bar', ax=ax4, color='steelblue', alpha=0.8)
        ax4.set_title('年化收益率', fontsize=12)
        ax4.set_ylabel('年化收益率')
        ax4.tick_params(axis='x', rotation=45, labelsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. 年化波动率
        ax5 = plt.subplot(3, 3, 5)
        annual_volatility.plot(kind='bar', ax=ax5, color='coral', alpha=0.8)
        ax5.set_title('年化波动率', fontsize=12)
        ax5.set_ylabel('年化波动率')
        ax5.tick_params(axis='x', rotation=45, labelsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. 夏普比率
        ax6 = plt.subplot(3, 3, 6)
        colors = ['green' if x > 0 else 'red' for x in sharpe_ratios]
        sharpe_ratios.plot(kind='bar', ax=ax6, color=colors, alpha=0.8)
        ax6.set_title('夏普比率', fontsize=12)
        ax6.set_ylabel('夏普比率')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.tick_params(axis='x', rotation=45, labelsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. 风险-收益散点图
        ax7 = plt.subplot(3, 3, 7)
        ax7.scatter(annual_volatility, annual_returns, s=100, alpha=0.7, c='steelblue')
        for i, symbol in enumerate(self.symbols):
            ax7.annotate(symbol, (annual_volatility.iloc[i], annual_returns.iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax7.set_xlabel('年化波动率')
        ax7.set_ylabel('年化收益率')
        ax7.set_title('风险-收益图', fontsize=12)
        ax7.grid(True, alpha=0.3)
        
        # 8. 30天滚动波动率
        ax8 = plt.subplot(3, 3, 8)
        rolling_vol = self.returns.rolling(window=30).std() * np.sqrt(252)
        rolling_vol.plot(ax=ax8, linewidth=1.5, alpha=0.8)
        ax8.set_title('30天滚动波动率', fontsize=12)
        ax8.set_ylabel('年化波动率')
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        ax8.grid(True, alpha=0.3)
        
        # 9. 累计收益对比
        ax9 = plt.subplot(3, 3, 9)
        cumulative_returns = (1 + self.returns).cumprod()
        cumulative_returns.plot(ax=ax9, linewidth=2, alpha=0.8)
        benchmark_cum = (1 + self.benchmark_returns).cumprod()
        ax9.plot(benchmark_cum.index, benchmark_cum.values, 
                linewidth=2, alpha=0.8, label=f'{self.benchmark}基准', color='black', linestyle='--')
        ax9.set_title('累计收益对比', fontsize=12)
        ax9.set_ylabel('累计收益')
        ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        ax9.grid(True, alpha=0.3)
        
        plt.suptitle('股票量化分析综合仪表板', fontsize=20, y=0.98)
        plt.tight_layout()
        plt.savefig('/workspace/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数 - 执行完整的量化分析流程"""
    print("=" * 80)
    print("股票量化分析系统")
    print("分析股票：OPEN, SERV, GLD, TLT, TQQQ, TSLA, META, NVDA")
    print("时间范围：2020年9月15日 - 2025年9月15日")
    print("=" * 80)
    
    # 定义股票代码和时间范围
    symbols = ['OPEN', 'SERV', 'GLD', 'TLT', 'TQQQ', 'TSLA', 'META', 'NVDA']
    start_date = '2020-09-15'
    end_date = '2025-09-15'
    
    # 创建分析器实例
    analyzer = QuantitativeAnalyzer(symbols, start_date, end_date)
    
    # 执行分析流程
    try:
        # 1. 获取数据
        if not analyzer.fetch_data():
            print("数据获取失败，程序退出")
            return
        
        # 2. 计算收益率
        analyzer.calculate_returns()
        
        # 3. 相关性分析
        correlation_matrix = analyzer.correlation_analysis()
        
        # 4. 主成分分析
        pca, components_df = analyzer.pca_analysis()
        
        # 5. 波动率分析
        annual_vol, rolling_vol = analyzer.volatility_analysis()
        
        # 6. 夏普比率分析
        sharpe_ratios, annual_returns, annual_volatility = analyzer.sharpe_ratio_analysis()
        
        # 7. Beta分析
        beta_series, alpha_series, r2_series = analyzer.beta_analysis()
        
        # 8. 投资组合策略
        strategy_stats, min_var_weights = analyzer.portfolio_strategies()
        
        # 9. 其他分析
        var_5, max_drawdown, rolling_corr_df = analyzer.additional_analysis()
        
        # 10. 综合仪表板
        analyzer.comprehensive_dashboard()
        
        print("\n" + "=" * 80)
        print("量化分析完成！所有图表已保存到工作目录")
        print("=" * 80)
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()