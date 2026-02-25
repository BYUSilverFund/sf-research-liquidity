import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv

from research.utils import run_backtest_parallel

# Load environment variables
load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "liquidity"
signal_name_title = "Liquidity"
columns = [
    "date",
    "barrid",
    "ticker",
    "return",
    "price",
    "daily_volume",
    "specific_return",
    "specific_risk",
    "predicted_beta",
]
IC = 0.05
gamma = 6000
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_10")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Load data
data = (
    sfd.load_assets(start=start, end=end, columns=columns, in_universe=True)
    .sort("barrid", "date")
    .with_columns(
        pl.col("price").shift(1).over("barrid").alias("price_lag"),
        pl.col("date").dt.truncate("1mo").alias("month"),
    )
    .with_columns(
        np.abs(pl.col("return"))
        .truediv(pl.col("daily_volume").mul(pl.col("price_lag")))
        .alias("raw_liq")
    )
)

# Calculate monthly signal and shift it
monthly_agg = (
    data.group_by(["barrid", "month"])
    .agg(
        pl.col("raw_liq").mean().alias(signal_name),
        pl.col("raw_liq").is_not_null().cast(pl.Int32).sum().alias("n_days"),
    )
    .sort(["barrid", "month"])
    .with_columns(
        pl.col(signal_name).shift(1).over("barrid"),
        pl.col("n_days").shift(1).over("barrid").alias("n_days_lag"),
    )
)

# Broadcast the monthly shifted signal back to the daily rows
signal = data.join(monthly_agg, on=["barrid", "month"], how="left")

# Filter data
filtered = signal.filter(
    pl.col("price_lag").gt(price_filter),
    pl.col(signal_name).is_finite(),
    pl.col("n_days_lag") >= 15,
    pl.col("specific_risk").is_not_null(),
    pl.col("predicted_beta").is_not_null(),
)

# Calculate scores
scores = filtered.select(
    "date",
    "barrid",
    "predicted_beta",
    "specific_risk",
    pl.col(signal_name)
    .sub(pl.col(signal_name).mean())
    .truediv(pl.col(signal_name).std())
    .over("date")
    .alias("score"),
)

# Compute alphas
alphas = (
    scores.with_columns(pl.col("score").mul(IC).mul("specific_risk").alias("alpha"))
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

# Get forward returns
forward_returns = (
    signal.sort("date", "barrid")
    .select(
        "date", "barrid", pl.col("return").shift(-1).over("barrid").alias("fwd_return")
    )
    .drop_nulls("fwd_return")
)

# Merge alphas and forward returns
merged = alphas.join(other=forward_returns, on=["date", "barrid"], how="inner")

# Get merged alphas and forward returns
merged_alphas = merged.select("date", "barrid", "alpha")
merged_forward_returns = merged.select("date", "barrid", "fwd_return")

# Get ics
ics = sfp.generate_alpha_ics(
    alphas=alphas, rets=forward_returns, method="rank", window=22
)

# Save ic chart
rank_chart_path = results_folder / "rank_ic_chart.png"
pearson_chart_path = results_folder / "pearson_ic_chart.png"

sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name_title} Cumulative IC",
    ic_type="Rank",
    file_name=rank_chart_path,
)
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name_title} Cumulative IC",
    ic_type="Pearson",
    file_name=pearson_chart_path,
)

# Run parallelized backtest
run_backtest_parallel(
    data=alphas,
    signal_name=signal_name,
    constraints=constraints,
    gamma=gamma,
    n_cpus=n_cpus,
)
