#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
美股多资产相关性与量化分析（2020-09-15 至 2025-09-15）
标的：OPEN, SERV, GLD, TLT, TQQQ, TSLA, META, NVDA；基准：SPY。
数据源：Yahoo Finance（真实历史数据）。

脚本功能：
- 下载每日复权收盘价（Adj Close），对齐并处理缺失
- 收盘价走势、收益率分布
- 收益率相关性热力图（皮尔逊）
- PCA主成分分析（解释方差与载荷）
- 波动率（年化、滚动）
- Sharpe Ratio（r_f=0）
- Beta（相对SPY，OLS与滚动Beta）
- 策略回测：等权、最小方差、动量
- 补充：最大回撤、简易有效前沿

所有图表保存到 /workspace/figures。
若个别标的在该区间无数据，将打印提示并从分析中剔除。
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # 无显示环境下保存图片

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import statsmodels.api as sm

TRADING_DAYS = 252
FIG_DIR = "/workspace/figures"
np.random.seed(42)
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.unicode_minus"] = False

TICKERS = ["OPEN", "SERV", "GLD", "TLT", "TQQQ", "TSLA", "META", "NVDA"]
BENCHMARK = "SPY"
START = "2020-09-15"
END = "2025-09-15"


def ensure_dir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def annualize_vol(daily_returns: pd.Series | pd.DataFrame):
	return daily_returns.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(series: pd.Series) -> float:
	cummax = series.cummax()
	dd = series / cummax - 1.0
	return float(dd.min())


def download_prices(tickers: list[str], benchmark: str, start: str, end: str):
	raw = yf.download(tickers + [benchmark], start=start, end=end, interval="1d", group_by="ticker", auto_adjust=False, progress=False)
	price_dict, available, missing = {}, [], []
	for t in tickers + [benchmark]:
		try:
			series = None
			if isinstance(raw.columns, pd.MultiIndex):
				if (t, "Adj Close") in raw.columns:
					series = raw[(t, "Adj Close")].rename(t)
				elif "Adj Close" in raw.columns.get_level_values(-1):
					series = raw.xs("Adj Close", level=-1, axis=1)[t].rename(t)
			else:
				if "Adj Close" in raw.columns:
					series = raw["Adj Close"].rename(t)
			if series is not None and series.dropna().shape[0] > 5:
				price_dict[t] = series
				available.append(t)
			else:
				missing.append(t)
		except Exception:
			missing.append(t)
	if not price_dict:
		raise RuntimeError("未能下载到任何有效的历史价格数据。")
	prices = pd.concat(price_dict.values(), axis=1).sort_index()
	prices = prices[~prices.index.duplicated(keep="first")].ffill()
	return prices, available, missing


def plot_and_save(ax, title: str, fname: str) -> None:
	ax.set_title(title)
	plt.tight_layout()
	plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
	plt.close()


def main() -> None:
	ensure_dir(FIG_DIR)
	print("图像输出目录:", FIG_DIR)

	prices, available, missing = download_prices(TICKERS, BENCHMARK, START, END)
	if missing:
		print("缺少或无效（已剔除）:", missing)

	valid_assets = [t for t in available if t != BENCHMARK]
	returns = prices[valid_assets].pct_change().dropna()
	market_returns = prices[BENCHMARK].pct_change().dropna() if BENCHMARK in prices.columns else None
	if market_returns is not None:
		returns = returns.loc[returns.index.intersection(market_returns.index)]
		market_returns = market_returns.loc[returns.index]
	print("有效资产数:", len(valid_assets), "数据起止:", returns.index.min().date(), "->", returns.index.max().date())

	# 1) 标准化价格
	norm_prices = prices[valid_assets].dropna()
	norm_prices = norm_prices / norm_prices.iloc[0]
	ax = norm_prices.plot(alpha=0.9)
	ax.set_xlabel("日期")
	ax.set_ylabel("标准化价格")
	plot_and_save(ax, "标准化收盘价走势（首日=1）", "01_norm_prices.png")

	# 2) 收益率分布
	ncols = 3
	nrows = len(valid_assets) // ncols + (1 if len(valid_assets) % ncols else 0)
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4*nrows))
	axes = axes.flatten()
	for i, t in enumerate(valid_assets):
		sns.histplot(returns[t].dropna(), bins=50, kde=True, ax=axes[i])
		axes[i].set_title(f"{t} 日收益率分布")
		axes[i].set_xlabel("日收益率")
		axes[i].set_ylabel("频数")
	for j in range(i+1, len(axes)):
		axes[j].axis("off")
	plt.tight_layout()
	plt.savefig(os.path.join(FIG_DIR, "02_return_distributions.png"), dpi=150)
	plt.close()

	# 3) 相关性热力图
	corr = returns.corr(method="pearson")
	plt.figure(figsize=(10, 8))
	sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=.5)
	plt.title("资产日收益率皮尔逊相关系数热力图")
	plt.tight_layout()
	plt.savefig(os.path.join(FIG_DIR, "03_corr_heatmap.png"), dpi=150)
	plt.close()

	# 4) PCA
	scaler = StandardScaler(with_mean=True, with_std=True)
	X = scaler.fit_transform(returns.values)
	n_components = min(5, len(valid_assets))
	pca = PCA(n_components=n_components, random_state=42)
	pca.fit(X)

	explained_ratio = pca.explained_variance_ratio_
	plt.figure(figsize=(10, 4))
	plt.bar(range(1, n_components+1), explained_ratio, color="C0")
	plt.plot(range(1, n_components+1), np.cumsum(explained_ratio), color="C1", marker="o", label="累计解释率")
	plt.xticks(range(1, n_components+1))
	plt.xlabel("主成分")
	plt.ylabel("解释方差比")
	plt.legend()
	plt.title("PCA解释方差比与累计解释率")
	plt.tight_layout()
	plt.savefig(os.path.join(FIG_DIR, "04_pca_explained_variance.png"), dpi=150)
	plt.close()

	loadings = pd.DataFrame(pca.components_.T, index=valid_assets, columns=[f"PC{i+1}" for i in range(n_components)])
	plt.figure(figsize=(12, 6))
	sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0)
	plt.title("PCA载荷（各资产在主成分上的权重）")
	plt.tight_layout()
	plt.savefig(os.path.join(FIG_DIR, "05_pca_loadings.png"), dpi=150)
	plt.close()

	# 5) 波动率
	ann_vol = annualize_vol(returns).sort_values(ascending=False)
	ax = ann_vol.plot(kind="bar", color="C2")
	ax.set_ylabel("年化波动率")
	plot_and_save(ax, "各资产年化波动率（按日收益率估计）", "06_annualized_vol.png")
	for window in [20, 60, 120]:
		roll_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
		ax = roll_vol.plot(alpha=0.85)
		ax.set_xlabel("日期")
		ax.set_ylabel("滚动年化波动率")
		plot_and_save(ax, f"{window}日滚动年化波动率", f"07_roll_vol_{window}d.png")

	# 6) Sharpe
	ann_return = returns.mean() * TRADING_DAYS
	ann_sharpe = ann_return / (ann_vol + 1e-12)
	sharpe_df = pd.DataFrame({"年化收益": ann_return, "年化波动率": ann_vol, "Sharpe": ann_sharpe}).sort_values("Sharpe", ascending=False)
	ax = sharpe_df["Sharpe"].plot(kind="bar", color="C1")
	ax.set_ylabel("Sharpe Ratio")
	plot_and_save(ax, "各资产年化Sharpe（r_f=0）", "08_sharpe.png")

	# 7) Beta
	if market_returns is not None:
		betas, alphas = {}, {}
		for t in valid_assets:
			X = sm.add_constant(market_returns.values)
			y = returns[t].values
			model = sm.OLS(y, X, missing="drop").fit()
			alphas[t] = float(model.params[0])
			betas[t] = float(model.params[1])
		beta_series = pd.Series(betas).sort_values(ascending=False)
		ax = beta_series.plot(kind="bar", color="C3")
		ax.set_ylabel("Beta 相对 SPY")
		plot_and_save(ax, "静态OLS估计的Beta（相对SPY）", "09_beta_static.png")

		window = 60
		roll_var_m = market_returns.rolling(window).var()
		roll_beta = {}
		for t in valid_assets:
			cov_im = returns[t].rolling(window).cov(market_returns)
			roll_beta[t] = cov_im / (roll_var_m + 1e-18)
		roll_beta_df = pd.DataFrame(roll_beta).dropna(how="all")
		ax = roll_beta_df.plot(alpha=0.9)
		ax.set_xlabel("日期")
		ax.set_ylabel("滚动Beta")
		plot_and_save(ax, f"{window}日滚动Beta（相对SPY）", "10_beta_rolling.png")
	else:
		print("未获取到基准SPY收益率，跳过Beta估计与绘图。")

	# 8) 策略
	def portfolio_nav(weights, ret_df: pd.DataFrame):
		if isinstance(weights, np.ndarray):
			port_ret = (ret_df * weights).sum(axis=1)
		else:
			port_ret = (ret_df * weights).sum(axis=1)
		nav = (1 + port_ret).cumprod()
		return nav, port_ret

	n = len(valid_assets)
	nav_df = pd.DataFrame(index=returns.index)
	if n > 0:
		w_eq = np.repeat(1.0 / n, n)
		nav_eq, _ = portfolio_nav(w_eq, returns[valid_assets])
		nav_df["等权"] = nav_eq

	def min_var_weights(ret_df: pd.DataFrame) -> np.ndarray:
		S = LedoitWolf().fit(ret_df.values).covariance_
		n = ret_df.shape[1]
		x0 = np.repeat(1.0 / n, n)
		def obj(w):
			return float(w.T @ S @ w)
		cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
		bounds = [(0.0, 1.0) for _ in range(n)]
		res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
		return res.x

	if n > 0:
		w_mv = min_var_weights(returns[valid_assets])
		nav_mv, _ = portfolio_nav(w_mv, returns[valid_assets])
		nav_df["最小方差"] = nav_mv

	lookback, topk = 60, min(3, n if n>0 else 3)
	weights_mom = pd.DataFrame(0.0, index=returns.index, columns=valid_assets)
	month_ends = returns.index.to_period("M").drop_duplicates().to_timestamp("M")
	month_ends = [d for d in month_ends if d in returns.index]
	for dt in month_ends:
		loc = returns.index.get_loc(dt)
		hist_start = loc - lookback
		if hist_start <= 0:
			continue
		hist_window = returns.iloc[hist_start:loc]
		momentum = (1 + hist_window).prod() - 1.0
		winners = momentum.sort_values(ascending=False).head(topk).index
		weights_mom.loc[dt, winners] = 1.0 / len(winners)
	weights_mom = weights_mom.replace(0, np.nan).ffill().fillna(0.0)
	if n > 0:
		nav_mom, _ = portfolio_nav(weights_mom, returns[valid_assets])
		nav_df["动量"] = nav_mom

	ax = nav_df.plot(alpha=0.9)
	ax.set_xlabel("日期")
	ax.set_ylabel("净值")
	plot_and_save(ax, "组合净值对比", "11_portfolio_nav.png")

	perf = {}
	for name, series in nav_df.items():
		r = series.pct_change().dropna()
		perf[name] = {
			"年化收益": float(r.mean()*TRADING_DAYS),
			"年化波动": float(r.std()*np.sqrt(TRADING_DAYS)),
			"Sharpe": float((r.mean()*TRADING_DAYS)/(r.std()*np.sqrt(TRADING_DAYS)+1e-12)),
			"最大回撤": max_drawdown(series)
		}
	perf_df = pd.DataFrame(perf).T.sort_values("Sharpe", ascending=False)
	perf_df.to_csv(os.path.join(FIG_DIR, "perf_summary.csv"), encoding="utf-8-sig")

	asset_nav = (1 + returns).cumprod()
	ax = asset_nav.plot(alpha=0.8)
	ax.set_xlabel("日期")
	ax.set_ylabel("净值")
	plot_and_save(ax, "单资产净值曲线", "12_asset_nav.png")

	if "等权" in nav_df.columns:
		series = nav_df["等权"]
		dd = series / series.cummax() - 1.0
		ax = dd.plot()
		ax.set_xlabel("日期")
		ax.set_ylabel("回撤")
		plot_and_save(ax, "等权组合回撤曲线", "13_drawdown_eq.png")

	if n > 0:
		M = 4000
		rand_weights = np.random.dirichlet(alpha=np.ones(n), size=M)
		port_returns, port_vols = [], []
		for w in rand_weights:
			r = (returns[valid_assets] * w).sum(axis=1)
			port_returns.append(float(r.mean()*TRADING_DAYS))
			port_vols.append(float(r.std()*np.sqrt(TRADING_DAYS)))
		plt.figure(figsize=(8, 6))
		sh = np.array(port_returns) / (np.array(port_vols) + 1e-12)
		plt.scatter(port_vols, port_returns, c=sh, cmap="viridis", s=8)
		plt.colorbar(label="Sharpe")
		plt.xlabel("年化波动率")
		plt.ylabel("年化收益率")
		plt.title("简易有效前沿（随机权重采样）")
		plt.tight_layout()
		plt.savefig(os.path.join(FIG_DIR, "14_efficient_frontier.png"), dpi=150)
		plt.close()

	print("图表与结果已生成，目录:", FIG_DIR)


if __name__ == "__main__":
	main()