import nbformat as nbf


def make_md(text: str):
    return nbf.v4.new_markdown_cell(text)


def make_code(code: str):
    return nbf.v4.new_code_cell(code)


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {
    "name": "python",
    "pygments_lexer": "ipython3",
}

cells = []

# Title and overview
cells.append(make_md(
    "# 量化分析：OPEN, SERV, GLD, TLT, TQQQ, TSLA, META, NVDA（2020-09-15 至 2025-09-15）\n\n"
    "本笔记使用 Yahoo Finance 的真实日度数据，对指定资产进行：\n"
    "- 收盘价与收益率分析\n"
    "- 收益率相关性（皮尔逊相关系数热力图）\n"
    "- 主成分分析（PCA）进行因子降维\n"
    "- 波动率分析（标准差与滚动波动率）\n"
    "- 风险调整收益（夏普比率，使用短端无风险利率近似）\n"
    "- Beta 系数与市场基准（SPY）关系（静态与滚动）\n"
    "- 策略模拟（等权重组合、最小方差组合、动量策略）\n"
    "- 其他补充（回撤、滚动相关性）\n\n"
    "说明：\n"
    "- 默认使用 Yahoo Finance真实数据；若某标的在区间内数据缺失或退市，将在输出中明确标注，并在相应计算中自动剔除或以可用数据进行。\n"
    "- 收益率基于“复权收盘价（Adj Close）”计算；展示价格时会标注。\n"
    "- 分析区间：2020-09-15 至 2025-09-15。\n"
))

# Imports and settings
cells.append(make_code(
    "# 导入依赖与全局设置（每个导入均附中文注释）\n"
    "import warnings  # 抑制不必要的警告\n"
    "warnings.filterwarnings('ignore')\n\n"
    "import pandas as pd  # 数据处理与时间序列分析\n"
    "import numpy as np   # 数值计算\n"
    "import yfinance as yf  # 从 Yahoo Finance 下载真实金融数据\n"
    "import seaborn as sns  # 统计可视化与热力图\n"
    "import matplotlib.pyplot as plt  # 通用绘图\n"
    "from sklearn.decomposition import PCA  # 主成分分析\n"
    "from sklearn.covariance import LedoitWolf  # 协方差收缩估计（稳健最小方差）\n"
    "from scipy.stats import linregress  # 线性回归（估计Beta等）\n\n"
    "# 让图像在Notebook内联显示\n"
    "%matplotlib inline\n\n"
    "# 统一绘图风格\n"
    "sns.set(style='whitegrid', context='talk')\n"
    "plt.rcParams['figure.figsize'] = (12, 6)\n"
    "plt.rcParams['axes.titlesize'] = 14\n"
    "plt.rcParams['axes.labelsize'] = 12\n\n"
    "# 定义分析区间与标的\n"
    "start_date = '2020-09-15'  # 起始日期\n"
    "end_date = '2025-09-15'    # 结束日期\n\n"
    "# 目标资产列表（用户指定）\n"
    "TICKERS = ['OPEN', 'SERV', 'GLD', 'TLT', 'TQQQ', 'TSLA', 'META', 'NVDA']\n"
    "# 市场基准与无风险近似（SPY：美股整体；^IRX：3M国债利率近似年化，无风险利率）\n"
    "BENCHMARK = 'SPY'\n"
    "RISK_FREE_PROXY = '^IRX'\n"
))

# Data download
cells.append(make_code(
    "# 封装一个下载函数：下载多资产的复权收盘价\n"
    "# 参数 tickers: 代码列表；start/end：日期范围\n"
    "# 返回：DataFrame，索引为日期，列为各资产的复权收盘价\n\n"
    "def download_adj_close(tickers, start, end):\n"
    "    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)['Adj Close']\n"
    "    # 若仅一个ticker时，返回Series，这里统一转换为DataFrame\n"
    "    if isinstance(data, pd.Series):\n"
    "        data = data.to_frame()\n"
    "    return data\n\n"
    "# 下载主资产价格、基准与无风险近似\n"
    "prices = download_adj_close(TICKERS, start_date, end_date)\n"
    "benchmark_prices = download_adj_close([BENCHMARK], start_date, end_date)\n"
    "risk_free_proxy = yf.download(RISK_FREE_PROXY, start=start_date, end=end_date, auto_adjust=False, progress=False)['Adj Close']\n\n"
    "# 处理缺失数据：报告缺失并前向填充/剔除\n"
    "missing_report = prices.isna().sum()\n"
    "print('各标的缺失值数量:\\n', missing_report)\n\n"
    "# 前向填充后再后向填充，最大允许2天缺失以平滑（更严可改为0）\n"
    "prices_ffill = prices.ffill(limit=2).bfill(limit=2)\n"
    "benchmark_prices_ffill = benchmark_prices.ffill(limit=2).bfill(limit=2)\n\n"
    "# 对于仍存在严重缺失的列（>10%），在后续分析中将剔除\n"
    "valid_cols = prices_ffill.columns[prices_ffill.isna().mean() < 0.1]\n"
    "prices_clean = prices_ffill[valid_cols]\n\n"
    "print('可用于后续分析的标的（缺失<10%）：', list(valid_cols))\n\n"
    "# 无风险利率：^IRX为年化百分比（近似），换算为日度（/100/252）\n"
    "rf_daily = (risk_free_proxy / 100.0) / 252.0\n"
    "rf_daily = rf_daily.reindex(prices_clean.index).ffill().bfill()\n\n"
    "# 保存中间结果，供后续复用\n"
    "prices_all = prices_clean.copy()\n"
    "benchmark_all = benchmark_prices_ffill.copy()\n"
))

# Missing tickers check and align
cells.append(make_code(
    "# 基础数据检查与清洗（标注缺失或不存在的代码）\n"
    "# 说明：有的代码可能不存在或在区间内无数据（如更名、退市），这里做显式检查。\n\n"
    "all_requested = set(TICKERS)\n"
    "all_downloaded = set(prices.columns)\n"
    "missing_tickers = sorted(list(all_requested - all_downloaded))\n"
    "if missing_tickers:\n"
    "    print('以下代码在Yahoo可能不存在或无法在该区间获取有效数据，将从分析中剔除：', missing_tickers)\n"
    "else:\n"
    "    print('所有请求的标的均已成功获取数据。')\n\n"
    "# 将有效列覆盖TICKERS顺序（剔除缺失者）\n"
    "TICKERS_VALID = [t for t in TICKERS if t in prices_all.columns]\n"
    "prices_all = prices_all[TICKERS_VALID]\n"
    "print('用于分析的有效标的：', TICKERS_VALID)\n\n"
    "# 与基准对齐索引\n"
    "benchmark_all = benchmark_all.reindex(prices_all.index).ffill().bfill()\n"
))

# Returns and normalized prices
cells.append(make_code(
    "# 计算日收益率、对数收益率，并绘制标准化价格走势\n"
    "# 收益率说明：使用复权收盘价的简单收益率 r_t = P_t / P_{t-1} - 1；对数收益率 ln(P_t / P_{t-1})\n\n"
    "# 计算简单收益率与对数收益率\n"
    "returns = prices_all.pct_change().dropna()\n"
    "log_returns = np.log(prices_all).diff().dropna()\n\n"
    "# 归一化价格（以起始值=1），用于多资产同图比较\n"
    "norm_prices = prices_all / prices_all.iloc[0]\n\n"
    "# 绘制：标准化价格随时间的走势\n"
    "ax = norm_prices.plot(title='标准化复权收盘价（起始=1）')\n"
    "ax.set_xlabel('日期')\n"
    "ax.set_ylabel('标准化价格')\n"
    "plt.show()\n\n"
    "# 简要统计描述（收益率）\n"
    "print('日度收益率描述性统计：')\n"
    "print(returns.describe().T[['mean','std','min','max']])\n"
))

# Correlation heatmap
cells.append(make_code(
    "# 相关性分析：皮尔逊相关系数与热力图\n"
    "# 计算收益率之间的相关性矩阵，并绘制热力图\n\n"
    "corr_matrix = returns.corr(method='pearson')\n"
    "plt.figure(figsize=(10, 8))\n"
    "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0)\n"
    "plt.title('日度收益率皮尔逊相关系数热力图')\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "# 将相关性矩阵保存\n"
    "corr_matrix.to_csv('/workspace/outputs/corr_matrix.csv')\n"
))

# PCA
cells.append(make_code(
    "# 主成分分析（PCA）：因子降维与解释方差\n"
    "# 使用收益率做PCA，先对列做去均值（可选标准化，这里直接使用收益率）\n\n"
    "# 去除含缺失的行\n"
    "returns_pca = returns.dropna(how='any').copy()\n\n"
    "# 拟合PCA（成分数不超过资产数）\n"
    "pca = PCA(n_components=min(returns_pca.shape[1], 6), random_state=42)\n"
    "pca_fit = pca.fit(returns_pca)\n\n"
    "# 解释方差比\n"
    "explained = pca.explained_variance_ratio_\n"
    "print('各主成分的解释方差比：', np.round(explained, 4))\n"
    "print('累计解释方差比：', np.round(np.cumsum(explained), 4))\n\n"
    "# 可视化：解释方差比柱状图与累计曲线\n"
    "fig, ax1 = plt.subplots(figsize=(10,5))\n"
    "components = np.arange(1, len(explained) + 1)\n"
    "ax1.bar(components, explained, color='steelblue', label='解释方差比')\n"
    "ax1.set_xlabel('主成分编号')\n"
    "ax1.set_ylabel('解释方差比')\n"
    "ax1.set_title('PCA 解释方差比与累计')\n"
    "ax2 = ax1.twinx()\n"
    "ax2.plot(components, np.cumsum(explained), color='orange', marker='o', label='累计解释方差比')\n"
    "ax2.set_ylabel('累计解释方差比')\n"
    "ax1.legend(loc='upper left')\n"
    "ax2.legend(loc='upper right')\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "# 主成分载荷（每个资产在各主成分上的权重）\n"
    "loadings = pd.DataFrame(pca.components_.T, index=returns_pca.columns, columns=[f'PC{i}' for i in range(1, len(explained)+1)])\n"
    "print('主成分载荷（前几列）：')\n"
    "print(loadings.head())\n\n"
    "# 可视化：前两个主成分载荷的散点图（风格图）\n"
    "plt.figure(figsize=(8,6))\n"
    "plt.scatter(loadings['PC1'], loadings['PC2'])\n"
    "for asset, row in loadings.iterrows():\n"
    "    plt.annotate(asset, (row['PC1'], row['PC2']))\n"
    "plt.axhline(0, color='gray', lw=1)\n"
    "plt.axvline(0, color='gray', lw=1)\n"
    "plt.title('资产在PC1-PC2空间的载荷分布')\n"
    "plt.xlabel('PC1载荷')\n"
    "plt.ylabel('PC2载荷')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# Volatility
cells.append(make_code(
    "# 波动率分析：标准差与滚动波动率\n"
    "# 年化波动率 ≈ 日度标准差 * sqrt(252)\n\n"
    "annualized_vol = returns.std() * np.sqrt(252)\n"
    "print('年化波动率（按日度std年化）：\\n', annualized_vol.sort_values(ascending=False))\n\n"
    "# 计算滚动波动率（如20日、60日），并绘制\n"
    "rolling_windows = [20, 60]\n"
    "fig, axes = plt.subplots(len(rolling_windows), 1, figsize=(12, 6*len(rolling_windows)), sharex=True)\n"
    "if len(rolling_windows) == 1:\n"
    "    axes = [axes]\n"
    "for ax, w in zip(axes, rolling_windows):\n"
    "    rolling_vol = returns.rolling(window=w).std() * np.sqrt(252)\n"
    "    rolling_vol.plot(ax=ax, title=f'{w}日滚动年化波动率')\n"
    "    ax.set_ylabel('年化波动率')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# Sharpe ratio
cells.append(make_code(
    "# 夏普比率：使用日度数据年化（252天），无风险利率来自^IRX（若缺失则置0）\n"
    "# 夏普 = (年化收益 - 年化无风险) / 年化波动率\n\n"
    "# 计算资产与基准的日度超额收益\n"
    "rf_daily_aligned = rf_daily.reindex(returns.index).fillna(0.0)\n"
    "excess_returns = returns.sub(rf_daily_aligned, axis=0)\n\n"
    "# 年化收益（简单近似）：mean * 252； 年化波动率：std * sqrt(252)\n"
    "ann_return = returns.mean() * 252\n"
    "ann_vol = returns.std() * np.sqrt(252)\n"
    "ann_rf = rf_daily_aligned.mean() * 252\n\n"
    "sharpe = (ann_return - ann_rf) / ann_vol\n\n"
    "print('年化收益(%)：\\n', (ann_return*100).round(2))\n"
    "print('年化波动率(%)：\\n', (ann_vol*100).round(2))\n"
    "print('夏普比率：\\n', sharpe.round(2))\n\n"
    "# 可视化：夏普比率柱状图\n"
    "plt.figure(figsize=(10,6))\n"
    "sharpe.sort_values(ascending=False).plot(kind='bar', color='teal', title='夏普比率（年化）')\n"
    "plt.ylabel('Sharpe Ratio')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# Beta vs SPY
cells.append(make_code(
    "# Beta 系数：相对于SPY（市场基准）\n"
    "# 方法1：静态Beta（全样本线性回归）；方法2：滚动Beta（例如60日窗口）\n\n"
    "# 计算基准日收益率\n"
    "benchmark_returns = benchmark_all.pct_change().dropna()\n"
    "benchmark_returns = benchmark_returns.iloc[:, 0]  # 转为Series（SPY）\n\n"
    "# 对齐资产与基准的日期\n"
    "aligned = returns.join(benchmark_returns.to_frame('SPY'), how='inner')\n"
    "R = aligned[TICKERS_VALID]\n"
    "M = aligned['SPY']\n\n"
    "# 静态Beta估计（线性回归：R_i = alpha + beta * M + eps）\n"
    "static_beta = {}\n"
    "for col in R.columns:\n"
    "    valid = R[col].dropna().index.intersection(M.dropna().index)\n"
    "    slope, intercept, r_value, p_value, std_err = linregress(M.loc[valid], R[col].loc[valid])\n"
    "    static_beta[col] = slope\n\n"
    "static_beta = pd.Series(static_beta)\n"
    "print('静态Beta（相对于SPY）：\\n', static_beta.sort_values(ascending=False).round(2))\n\n"
    "# 滚动Beta（60日窗口）：beta_t = Cov(R_i, M) / Var(M)\n"
    "window = 60\n"
    "rolling_beta = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)\n"
    "var_m = M.rolling(window).var()\n"
    "for col in R.columns:\n"
    "    cov_im = R[col].rolling(window).cov(M)\n"
    "    rolling_beta[col] = cov_im / var_m\n\n"
    "# 可视化：滚动Beta\n"
    "fig, axes = plt.subplots(len(TICKERS_VALID), 1, figsize=(12, 3*len(TICKERS_VALID)), sharex=True)\n"
    "if len(TICKERS_VALID) == 1:\n"
    "    axes = [axes]\n"
    "for ax, col in zip(axes, TICKERS_VALID):\n"
    "    rolling_beta[col].plot(ax=ax, title=f'滚动Beta（{col} vs SPY, 窗口={window}日）')\n"
    "    ax.axhline(static_beta[col], color='red', linestyle='--', alpha=0.6, label='静态Beta')\n"
    "    ax.legend(loc='upper right')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# Strategies
cells.append(make_code(
    "# 策略模拟：等权重、最小方差、动量策略\n"
    "# - 等权重：每期持有等权重，月度再平衡。\n"
    "# - 最小方差：使用Ledoit-Wolf收缩协方差估计，最小方差权重（约束权重之和=1，非负投影）。\n"
    "# - 动量：过去252个交易日的动量排名，买入前N只（如3只），月度调仓。\n\n"
    "from datetime import datetime\n\n"
    "# 月末重采样（调仓点）\n"
    "rebalance_dates = returns.resample('M').last().index\n\n"
    "# 1) 等权重组合（每月初等权再平衡）\n"
    "weights_eq = pd.DataFrame(0.0, index=rebalance_dates, columns=TICKERS_VALID)\n"
    "weights_eq.loc[:, :] = 1.0 / len(TICKERS_VALID)\n\n"
    "# 将权重向前填充到当月每日\n"
    "w_eq_daily = weights_eq.reindex(returns.index, method='ffill').fillna(method='ffill')\n"
    "port_eq = (w_eq_daily * returns).sum(axis=1)\n\n"
    "# 2) 最小方差组合（每月用过去60日协方差估计）\n"
    "lookback_mv = 60\n"
    "weights_mv = pd.DataFrame(0.0, index=rebalance_dates, columns=TICKERS_VALID)\n"
    "for d in rebalance_dates:\n"
    "    window_returns = returns.loc[returns.index <= d].tail(lookback_mv)\n"
    "    if window_returns.shape[0] < lookback_mv:\n"
    "        continue\n"
    "    # Ledoit-Wolf协方差估计\n"
    "    lw = LedoitWolf().fit(window_returns.values)\n"
    "    cov = lw.covariance_\n"
    "    # 最小方差权重闭式解并投影到非负\n"
    "    ones = np.ones((cov.shape[0], 1))\n"
    "    inv_cov = np.linalg.pinv(cov)\n"
    "    raw_w = inv_cov @ ones\n"
    "    raw_w = raw_w / (ones.T @ inv_cov @ ones)\n"
    "    w = np.maximum(raw_w.flatten(), 0)\n"
    "    w = w / w.sum() if w.sum() > 0 else np.ones_like(w)/len(w)\n"
    "    weights_mv.loc[d, :] = w\n\n"
    "w_mv_daily = weights_mv.reindex(returns.index, method='ffill').fillna(method='ffill')\n"
    "port_mv = (w_mv_daily * returns).sum(axis=1)\n\n"
    "# 3) 动量策略：过去252日累计收益，选前3只，月度换仓\n"
    "lookback_mom = 252\n"
    "top_n = 3\n"
    "weights_mom = pd.DataFrame(0.0, index=rebalance_dates, columns=TICKERS_VALID)\n"
    "for d in rebalance_dates:\n"
    "    window_returns = returns.loc[:d].tail(lookback_mom)\n"
    "    if window_returns.shape[0] < lookback_mom:\n"
    "        continue\n"
    "    cumret = (1 + window_returns).prod() - 1\n"
    "    winners = cumret.sort_values(ascending=False).head(top_n).index\n"
    "    weights_mom.loc[d, winners] = 1.0 / top_n\n\n"
    "w_mom_daily = weights_mom.reindex(returns.index, method='ffill').fillna(0.0)\n"
    "port_mom = (w_mom_daily * returns).sum(axis=1)\n\n"
    "# 汇总组合净值并可视化\n"
    "portfolio_nav = pd.DataFrame({\n"
    "    'EQ': (1 + port_eq).cumprod(),\n"
    "    'MinVar': (1 + port_mv).cumprod(),\n"
    "    'Momentum': (1 + port_mom).cumprod(),\n"
    "    'SPY': (1 + benchmark_returns.reindex(returns.index, method='ffill')).cumprod()\n"
    "}, index=returns.index)\n\n"
    "ax = portfolio_nav.plot(title='策略净值对比（起始=1）')\n"
    "ax.set_ylabel('净值')\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "# 输出策略绩效（年化收益、波动、夏普、最大回撤）\n"
    "def max_drawdown(nav_series: pd.Series) -> float:\n"
    "    cummax = nav_series.cummax()\n"
    "    dd = nav_series / cummax - 1\n"
    "    return dd.min()\n\n"
    "perf = {}\n"
    "for name, nav in portfolio_nav.items():\n"
    "    ret = nav.pct_change().dropna()\n"
    "    ann_ret = ret.mean() * 252\n"
    "    ann_vol = ret.std() * np.sqrt(252)\n"
    "    sharpe_p = (ann_ret - ann_rf) / (ann_vol + 1e-9)\n"
    "    mdd = max_drawdown(nav)\n"
    "    perf[name] = {\n"
    "        'AnnReturn': ann_ret,\n"
    "        'AnnVol': ann_vol,\n"
    "        'Sharpe': sharpe_p,\n"
    "        'MaxDD': mdd,\n"
    "    }\n\n"
    "perf_df = pd.DataFrame(perf).T\n"
    "print('策略绩效摘要：\\n', perf_df.round(3))\n"
))

# Additional analyses
cells.append(make_code(
    "# 其他分析：最大回撤曲线与滚动相关性\n\n"
    "# 1) 单资产净值与回撤曲线\n"
    "asset_nav = (1 + returns).cumprod()\n"
    "asset_rolling_max = asset_nav.cummax()\n"
    "asset_drawdown = asset_nav / asset_rolling_max - 1\n\n"
    "fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)\n"
    "asset_nav.plot(ax=axes[0], title='单资产净值（起始=1）')\n"
    "axes[0].set_ylabel('净值')\n"
    "asset_drawdown.plot(ax=axes[1], title='单资产回撤（Drawdown）')\n"
    "axes[1].set_ylabel('回撤')\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "# 2) 滚动相关性（与SPY的60日滚动相关）\n"
    "rolling_corr = pd.DataFrame(index=returns.index, columns=TICKERS_VALID)\n"
    "for col in TICKERS_VALID:\n"
    "    rolling_corr[col] = returns[col].rolling(60).corr(benchmark_returns.reindex(returns.index))\n\n"
    "ax = rolling_corr.plot(title='与SPY的60日滚动相关性')\n"
    "ax.set_ylabel('相关系数')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# Export plots and tables
cells.append(make_code(
    "# 图表导出（可选）：将关键图表与表格保存到 /workspace/outputs 目录\n"
    "# 说明：在交互式环境中图表已内联显示，这里再存为文件以便报告复用\n\n"
    "import os\n"
    "os.makedirs('/workspace/outputs', exist_ok=True)\n\n"
    "# 示例：保存净值图\n"
    "ax = portfolio_nav.plot(title='策略净值对比（起始=1）')\n"
    "ax.set_ylabel('净值')\n"
    "plt.tight_layout()\n"
    "plt.savefig('/workspace/outputs/strategy_nav.png', dpi=150)\n"
    "plt.close()\n\n"
    "# 保存夏普比率柱状图\n"
    "plt.figure(figsize=(10,6))\n"
    "sharpe.sort_values(ascending=False).plot(kind='bar', color='teal', title='夏普比率（年化）')\n"
    "plt.ylabel('Sharpe Ratio')\n"
    "plt.tight_layout()\n"
    "plt.savefig('/workspace/outputs/sharpe_bar.png', dpi=150)\n"
    "plt.close()\n\n"
    "# 保存相关性矩阵csv已在前面完成\n"
    "# 可选保存载荷矩阵\n"
    "loadings.to_csv('/workspace/outputs/pca_loadings.csv')\n\n"
    "# 保存策略绩效表\n"
    "perf_df.round(3).to_csv('/workspace/outputs/strategy_performance.csv')\n"
    "print('图表与表格已导出到 /workspace/outputs')\n"
))

# Run instructions
cells.append(make_md(
    "## 运行与复现说明\n\n"
    "- 环境依赖：`yfinance`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `notebook`。\n"
    "- 数据来源：Yahoo Finance（公开API）。本笔记会直接在线抓取 2020-09-15 至 2025-09-15 的真实数据；若单个标的在该区间无有效数据（例如 `SERV`），上文会打印通知并自动剔除，不参与计算。\n"
    "- 运行方式：\n"
    "  1. 在终端执行 `jupyter notebook` 打开此笔记。\n"
    "  2. 逐个运行所有单元格。\n"
    "- 输出：\n"
    "  - 图表在Notebook内联展示，同时部分图与表格导出至 `/workspace/outputs`。\n"
    "- 注意：\n"
    "  - Yahoo的历史数据会有偶发性缺失或更名，已在代码中进行前填充/后填充与有效列筛选（缺失占比<10%），并对缺失超限的标的进行剔除。\n"
    "  - 无风险利率采用 `^IRX`（3M T-Bill 近似）换算为日度；若该序列缺失则回退为0。\n"
))

nb["cells"] = cells

import os
os.makedirs("/workspace/notebooks", exist_ok=True)
with open("/workspace/notebooks/quant_analysis_yahoo_2020_2025.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook rebuilt at /workspace/notebooks/quant_analysis_yahoo_2020_2025.ipynb")

