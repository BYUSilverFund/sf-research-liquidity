import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import sf_quant.data as sfd
import statsmodels.formula.api as smf

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "liquidity"
gamma = 6000
results_folder = Path("results/experiment_10")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Load MVO weights
weights = pl.read_parquet(f"weights/{signal_name}/{gamma}/*.parquet")

# Get returns
returns = (
    sfd.load_assets(
        start=start, end=end, columns=["date", "barrid", "return"], in_universe=True
    )
    .sort("date", "barrid")
    .select(
        "date",
        "barrid",
        pl.col("return").truediv(100).shift(-1).over("barrid").alias("forward_return"),
    )
)

# Compute portfolio returns
portfolio_returns = (
    weights.join(other=returns, on=["date", "barrid"], how="left")
    .group_by("date")
    .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("return"))
    .sort("date")
)

# Compute cumulative log returns
cumulative_returns = portfolio_returns.select(
    "date", pl.col("return").log1p().cum_sum().mul(100).alias("cumulative_return")
)

# Plot cumulative log returns
plt.figure(figsize=(10, 5))
plt.plot(cumulative_returns["date"], cumulative_returns["cumulative_return"])
plt.title("MVO Backtest Results (Active)")
plt.ylabel("Cumulative Log Return (%)")
plt.grid(True, alpha=0.3)

chart_path = results_folder / "cumulative_returns.png"
plt.savefig(chart_path, bbox_inches="tight", dpi=300)
plt.close()

# Create summary table
summary = portfolio_returns.select(
    pl.col("return").mean().mul(252).alias("mean_return"),
    pl.col("return").std().mul(pl.lit(252).sqrt()).alias("volatility"),
).with_columns(pl.col("mean_return").truediv(pl.col("volatility")).alias("sharpe"))

# Save summary table with Matplotlib
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis("off")
ax.axis("tight")

mean_ret = summary["mean_return"][0]
vol = summary["volatility"][0]
sharpe = summary["sharpe"][0]

cell_text = [[f"{mean_ret:.2%}", f"{vol:.2%}", f"{sharpe:.2f}"]]
col_labels = ["Mean Return", "Volatility", "Sharpe"]

ax.set_title("MVO Backtest Results (Active)", loc="center", pad=15)
table = ax.table(
    cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center"
)
table.scale(1, 2)

table_path = results_folder / "summary_table.png"
plt.savefig(table_path, bbox_inches="tight", dpi=300)
plt.close()


# Fama french regression
ff5 = (
    sfd.load_fama_french(start=start, end=end)
    .sort("date")
    .with_columns(pl.exclude("date").shift(-1))
)

regression_data = (
    portfolio_returns.join(other=ff5, on="date", how="left")
    .drop_nulls("return")
    .with_columns(pl.col("return").sub("rf").alias("return_rf"))
    .with_columns(pl.exclude("date").mul(100))
)

formula = "return_rf ~ mkt_rf + smb + hml + rmw + cma"
model = smf.ols(formula, regression_data)
results = model.fit()

regression_summary = pl.DataFrame(
    {
        "variable": results.params.index,
        "coefficient": results.params.values,
        "tstat": results.tvalues.values,
    }
)

# Save regression table with Matplotlib
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis("off")
ax.axis("tight")

cell_text = [
    [row[0], f"{row[1]:.4f}", f"{row[2]:.4f}"] for row in regression_summary.iter_rows()
]
col_labels = ["Variable", "Coefficient", "T-stat"]

ax.set_title("MVO Backtest Results (Active) (Daily %)", loc="center", pad=15)
table = ax.table(
    cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center"
)
table.scale(1, 1.5)

table_path = results_folder / "regression_table.png"
plt.savefig(table_path, bbox_inches="tight", dpi=300)
plt.close()
