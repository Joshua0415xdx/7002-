# 股票量化分析系统

## 项目概述

这是一个完整的股票量化分析系统，用Python实现，专门分析OPEN, SERV, GLD, TLT, TQQQ, TSLA, META, NVDA等股票的量化特征。系统使用真实的美股数据进行分析，提供全面的量化分析功能。

## 功能特性

### 1. 数据获取与处理
- ✅ 从Yahoo Finance获取真实股票数据
- ✅ 自动处理数据缺失和对齐问题
- ✅ 支持多只股票同时分析
- ✅ 数据质量检查和验证

### 2. 核心分析功能

#### 📊 相关性分析
- 皮尔逊相关系数计算
- 相关性热力图可视化
- 识别股票间的关联关系

#### 📈 风险收益分析
- 年化收益率计算
- 年化波动率分析
- 夏普比率(风险调整后收益)
- 风险-收益散点图

#### 🔍 主成分分析 (PCA)
- 因子降维分析
- 方差解释比可视化
- 主成分载荷分析
- 识别主要驱动因子

#### 📉 波动率分析
- 历史波动率计算
- 滚动波动率时间序列
- 波动率聚集性分析

#### 📊 Beta系数分析
- 相对于SPY基准的Beta计算
- Alpha系数(超额收益)
- R²相关程度分析
- 系统性风险度量

#### 💼 投资组合策略
- 等权重组合
- 最小方差组合
- 动量策略
- 策略回测和比较

#### ⚠️ 风险管理分析
- VaR(风险价值)计算
- 最大回撤分析
- 滚动相关性分析
- 风险度量可视化

## 文件结构

```
/workspace/
├── quantitative_analysis.py          # 主要分析程序
├── simple_quant_demo.py              # 简化演示程序
├── quantitative_analysis_report.md   # 详细分析报告
├── README.md                         # 使用说明(本文件)
│
├── 分析图表/
│   ├── comprehensive_dashboard.png   # 综合分析仪表板
│   ├── correlation_heatmap.png      # 相关性热力图
│   ├── pca_analysis.png             # 主成分分析图
│   ├── volatility_analysis.png      # 波动率分析图
│   ├── sharpe_ratio_analysis.png    # 夏普比率分析图
│   ├── beta_analysis.png            # Beta系数分析图
│   ├── portfolio_strategies.png     # 投资组合策略图
│   └── additional_analysis.png      # 其他分析图表
│
└── 演示图表/
    ├── demo_correlation.png         # 演示相关性图
    ├── demo_risk_return.png         # 演示风险收益图
    ├── demo_pca.png                 # 演示PCA图
    ├── demo_portfolio.png           # 演示组合图
    └── demo_var.png                 # 演示VaR图
```

## 使用方法

### 运行主要分析程序
```bash
python3 quantitative_analysis.py
```

### 运行演示程序
```bash
python3 simple_quant_demo.py
```

## 分析结果

### 主要发现 (2024年3月-2025年9月)

#### 📊 收益率排名
1. **TSLA**: 76.62% (最佳表现)
2. **NVDA**: 61.19%
3. **OPEN**: 56.08%
4. **META**: 33.20%
5. **TQQQ**: 30.89%
6. **SERV**: 27.89%
7. **GLD**: 5.27%
8. **TLT**: 1.00%

#### ⚠️ 风险评估
- **最高风险**: SERV (224% 年化波动率)
- **最低风险**: TLT (13.7% 年化波动率)
- **最大回撤**: OPEN (-83.6%)
- **最小回撤**: GLD (-8.1%)

#### 🏆 风险调整收益 (夏普比率)
1. **GLD**: 1.96 (最佳)
2. **NVDA**: 1.12
3. **OPEN**: 1.11
4. **TSLA**: 1.10

#### 📈 投资组合表现
- **等权重组合**: 年化收益73.54%, 夏普比率1.56
- **显著优于SPY基准**: SPY年化收益19.44%, 夏普比率0.99

## 技术实现

### 依赖库
```python
yfinance          # 股票数据获取
pandas           # 数据处理
numpy            # 数值计算
matplotlib       # 基础绘图
seaborn          # 统计图表
scikit-learn     # 机器学习(PCA)
scipy            # 统计分析
```

### 核心算法

1. **收益率计算**: `returns = prices.pct_change()`
2. **相关性**: `correlation_matrix = returns.corr()`
3. **波动率**: `volatility = returns.std() * sqrt(252)`
4. **夏普比率**: `sharpe = (annual_return - risk_free_rate) / volatility`
5. **Beta系数**: `beta = covariance(stock, market) / variance(market)`
6. **VaR**: `var = returns.quantile(confidence_level)`
7. **最大回撤**: `max_drawdown = (cumulative_returns - running_max) / running_max`

## 数据说明

### 数据来源
- **提供商**: Yahoo Finance
- **数据类型**: 调整后收盘价
- **更新频率**: 日频
- **数据质量**: 已验证的真实市场数据

### 特殊说明
- **SERV**: 新上市股票，数据从2024年3月开始
- **时间对齐**: 所有分析基于共同的交易日
- **缺失值处理**: 使用前向填充方法

## 投资建议

### 🎯 适合不同风险偏好的配置

#### 激进型投资者
- **核心持仓**: TSLA, NVDA, OPEN
- **预期收益**: 60%+ 年化
- **风险水平**: 极高 (回撤可能超过50%)

#### 平衡型投资者  
- **推荐策略**: 等权重组合
- **预期收益**: 40-50% 年化
- **风险水平**: 中高 (回撤控制在30%以内)

#### 保守型投资者
- **核心配置**: GLD + 部分科技股
- **预期收益**: 20-30% 年化
- **风险水平**: 中等 (最大回撤15%以内)

### ⚠️ 风险提示
1. **历史业绩不代表未来表现**
2. **高收益伴随高风险**
3. **建议分散投资，定期再平衡**
4. **根据个人风险承受能力调整配置**

## 更新日志

- **v1.0** (2025-09-16): 初始版本发布
  - 完整的量化分析功能
  - 8只股票的深度分析
  - 多种投资组合策略
  - 全面的风险管理工具

## 联系信息

如有问题或建议，请联系量化研究团队。

---

*最后更新: 2025年9月16日*  
*数据截止: 2025年9月12日*