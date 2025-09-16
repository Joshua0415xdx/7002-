#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版量化分析演示程序
用于展示核心量化分析功能和方法
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

def get_stock_data(symbols, start_date, end_date):
    """获取股票数据 - 从Yahoo Finance获取真实股价数据"""
    print(f"正在获取股票数据: {', '.join(symbols)}")
    print(f"时间范围: {start_date} 到 {end_date}")
    
    # 获取股票数据
    data = yf.download(symbols, start=start_date, end=end_date, progress=False)
    
    # 提取收盘价
    if data.columns.nlevels > 1:
        prices = data['Close']
    else:
        prices = data
    
    print(f"成功获取 {len(prices.columns)} 只股票的数据")
    print(f"数据点数: {len(prices)} 个交易日")
    
    return prices

def calculate_returns(prices):
    """计算收益率 - 基于每日收盘价计算日收益率"""
    returns = prices.pct_change().dropna()
    print(f"\n收益率计算完成，共 {len(returns)} 个数据点")
    return returns

def correlation_analysis(returns):
    """相关性分析 - 计算并可视化皮尔逊相关系数"""
    print("\n进行相关性分析...")
    
    # 计算相关系数矩阵
    corr_matrix = returns.corr()
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, fmt='.3f')
    plt.title('股票收益率相关性热力图', fontsize=14)
    plt.tight_layout()
    plt.savefig('/workspace/demo_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix

def risk_return_analysis(returns):
    """风险收益分析 - 计算年化收益率、波动率和夏普比率"""
    print("\n进行风险收益分析...")
    
    # 计算关键指标
    annual_returns = returns.mean() * 252  # 年化收益率
    annual_volatility = returns.std() * np.sqrt(252)  # 年化波动率
    sharpe_ratios = (annual_returns - 0.02) / annual_volatility  # 夏普比率(假设无风险利率2%)
    
    # 创建结果DataFrame
    risk_return_df = pd.DataFrame({
        '年化收益率': annual_returns,
        '年化波动率': annual_volatility,
        '夏普比率': sharpe_ratios
    })
    
    print("风险收益统计:")
    print(risk_return_df.round(4))
    
    # 绘制风险-收益散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(annual_volatility, annual_returns, s=100, alpha=0.7, c='steelblue')
    
    # 添加股票标签
    for i, symbol in enumerate(returns.columns):
        plt.annotate(symbol, (annual_volatility.iloc[i], annual_returns.iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.xlabel('年化波动率 (风险)')
    plt.ylabel('年化收益率 (收益)')
    plt.title('风险-收益散点图', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/workspace/demo_risk_return.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return risk_return_df

def pca_analysis(returns):
    """主成分分析 - 进行因子降维分析"""
    print("\n进行主成分分析...")
    
    # 标准化数据
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns)
    
    # 进行PCA
    pca = PCA()
    pca_result = pca.fit_transform(returns_scaled)
    
    # 绘制方差解释比
    plt.figure(figsize=(12, 5))
    
    # 左图：方差解释比
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, alpha=0.7)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'ro-', linewidth=2)
    plt.xlabel('主成分')
    plt.ylabel('方差解释比')
    plt.title('PCA方差解释比')
    plt.grid(True, alpha=0.3)
    
    # 右图：前两个主成分散点图
    plt.subplot(1, 2, 2)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('前两个主成分散点图')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/demo_pca.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 显示主成分载荷
    components_df = pd.DataFrame(
        pca.components_[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=returns.columns
    )
    print("前3个主成分载荷:")
    print(components_df.round(4))
    
    return pca, components_df

def portfolio_analysis(returns):
    """投资组合分析 - 等权重组合vs个股表现"""
    print("\n进行投资组合分析...")
    
    # 等权重组合收益
    equal_weight_returns = returns.mean(axis=1)
    
    # 计算累计收益
    cumulative_returns = (1 + returns).cumprod()
    equal_weight_cumulative = (1 + equal_weight_returns).cumprod()
    
    # 绘制累计收益图
    plt.figure(figsize=(12, 8))
    
    # 个股累计收益
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column], 
                linewidth=2, alpha=0.8, label=column)
    
    # 等权重组合
    plt.plot(equal_weight_cumulative.index, equal_weight_cumulative, 
            linewidth=3, color='black', linestyle='--', label='等权重组合')
    
    plt.xlabel('日期')
    plt.ylabel('累计收益')
    plt.title('累计收益对比 (个股 vs 等权重组合)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/workspace/demo_portfolio.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 组合统计
    portfolio_annual_return = equal_weight_returns.mean() * 252
    portfolio_annual_vol = equal_weight_returns.std() * np.sqrt(252)
    portfolio_sharpe = (portfolio_annual_return - 0.02) / portfolio_annual_vol
    
    print(f"等权重组合统计:")
    print(f"年化收益率: {portfolio_annual_return:.2%}")
    print(f"年化波动率: {portfolio_annual_vol:.2%}")
    print(f"夏普比率: {portfolio_sharpe:.4f}")
    
    return equal_weight_returns

def value_at_risk(returns, confidence_level=0.05):
    """风险价值计算 - VaR分析"""
    print(f"\n计算 {confidence_level*100}% VaR...")
    
    var_values = returns.quantile(confidence_level)
    
    # 绘制VaR图表
    plt.figure(figsize=(10, 6))
    var_values.plot(kind='bar', color='red', alpha=0.8)
    plt.title(f'{confidence_level*100}% VaR (风险价值)', fontsize=14)
    plt.ylabel('VaR')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/workspace/demo_var.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("VaR值 (从高风险到低风险):")
    print(var_values.sort_values().round(4))
    
    return var_values

def main():
    """主函数 - 演示完整的量化分析流程"""
    print("=" * 60)
    print("量化分析演示程序")
    print("=" * 60)
    
    # 定义股票和时间范围
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']  # 使用常见股票进行演示
    start_date = '2020-01-01'
    end_date = '2025-01-01'
    
    try:
        # 1. 获取数据
        prices = get_stock_data(symbols, start_date, end_date)
        
        # 2. 计算收益率
        returns = calculate_returns(prices)
        
        # 3. 相关性分析
        corr_matrix = correlation_analysis(returns)
        
        # 4. 风险收益分析
        risk_return_df = risk_return_analysis(returns)
        
        # 5. 主成分分析
        pca, components = pca_analysis(returns)
        
        # 6. 投资组合分析
        portfolio_returns = portfolio_analysis(returns)
        
        # 7. 风险价值分析
        var_values = value_at_risk(returns)
        
        print("\n" + "=" * 60)
        print("量化分析演示完成！")
        print("生成的图表文件:")
        print("- demo_correlation.png: 相关性热力图")
        print("- demo_risk_return.png: 风险收益散点图") 
        print("- demo_pca.png: 主成分分析")
        print("- demo_portfolio.png: 投资组合分析")
        print("- demo_var.png: 风险价值分析")
        print("=" * 60)
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()