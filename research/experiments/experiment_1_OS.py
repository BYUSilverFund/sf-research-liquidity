import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
import sf_quant.data as sfd
from matplotlib import pyplot as plt

# Paramters (out of sample)
start = dt.date(2010, 1, 1)
end = dt.date(2024, 12, 31)
columns = ["date", "permno", "ret", "prc", "vol", "shrout"]
price_filter = 5
n_bins = 5
labels = [str(i) for i in range(n_bins)]
signal_name = "liquidity"
results_folder = Path("results/experiment_1")

results_folder.mkdir(parents=True, exist_ok=True)

# Data (out of sample)
data = (
    sfd.load_crsp_v2_daily(start=start, end=end, columns=columns)
    .with_columns(pl.col("prc").abs().alias("prc_abs"))
    .with_columns(
        pl.col("prc_abs").shift(1).over("permno").alias("price_lag"),
    )
    .with_columns(pl.col("date").dt.truncate("1mo"))
)

# Calculate signal
signal = (
    data.sort("permno", "date")
    .with_columns(
        np.abs(pl.col("ret"))
        .truediv(pl.col("vol").mul(pl.col("prc_abs")))
        .alias("liquidity")
    )
    .group_by(["permno", "date"])
    .agg(
        pl.col("liquidity").mean(),
        (pl.col("ret") + 1).product().sub(1).alias("ret"),
        pl.col("price_lag").last(),
        pl.len().alias("n_days"),
    )
    .sort(["permno", "date"])
    .with_columns(
        pl.col("liquidity").shift(1).over("permno"),
        pl.col("n_days").shift(1).over("permno").alias("n_days_lag"),
    )
)

# Filtering
filtered = signal.filter(
    # pl.col("price_lag").gt(price_filter),
    pl.col(signal_name).is_finite(),
    pl.col("n_days_lag") >= 15,
)

# Portfolios
portfolios = filtered.with_columns(
    pl.col(signal_name)
    .rank(method="ordinal")
    .qcut(n_bins, labels=labels, allow_duplicates=True)
    .over("date")
    .alias("bin")
)

returns = (
    portfolios.group_by("date", "bin")
    .agg(pl.col("ret").mean().alias("return"))
    .pivot(on="bin", index="date", values="return")
    .with_columns(pl.col(str(n_bins - 1)).sub(pl.col("0")).alias("spread"))
    .unpivot(index="date", variable_name="bin", value_name="return")
    .sort("date", "bin")
)

# Plotting
plot_df = (
    returns.sort("date", "bin")
    .select(
        "date",
        "bin",
        pl.col("return")
        .log1p()
        .cum_sum()
        .mul(100)
        .over("bin")
        .alias("cumulative_return"),
    )
    .pivot(on="bin", index="date", values="cumulative_return")
    .fill_null(strategy="forward")
)

plt.figure(figsize=(10, 6))
colors = sns.color_palette("coolwarm", n_bins).as_hex()
colors.append("green")

x_data = plot_df["date"].to_numpy()
bin_list = [str(j) for j in range(n_bins)] + ["spread"]

for i, label in enumerate(bin_list):
    if label in plot_df.columns:
        y_data = plot_df[label].to_numpy()
        mask = ~np.isnan(y_data)
        sns.lineplot(
            x=x_data[mask],
            y=y_data[mask],
            label=label,
            color=colors[i] if i < len(colors) else "black",
        )

plt.title("Liquidity Backtest: Out-of-Sample (2010-2024)")
plt.ylabel("Cumulative Log Return (%)")
plt.legend(title="Portfolio", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig(results_folder / "cumulative_returns_oos.png", dpi=300)

# Summary stats
summary = (
    returns.group_by("bin")
    .agg(
        mean_return=pl.col("return").mean().mul(12),
        volatility=pl.col("return").std().mul(np.sqrt(12)),
    )
    .with_columns((pl.col("mean_return").truediv(pl.col("volatility"))).alias("sharpe"))
    .sort("bin", descending=True)
)

plt.figure(figsize=(12, 6))
plt.axis("off")
table_data = [summary.columns] + summary.with_columns(
    pl.col(pl.Float64).round(4)
).to_numpy().tolist()
the_table = plt.table(
    cellText=table_data,
    loc="center",
    cellLoc="center",
    colWidths=[0.15] * len(summary.columns),
)
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1.2, 1.2)

# Save
plt.savefig(results_folder / "summary_table_oos.png", dpi=300, bbox_inches="tight")
