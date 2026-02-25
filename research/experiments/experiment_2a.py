# Amihud 2015 Replication for U.S. stocks only. (In sample 1990-2011. 
# This replication uses daily rebalances. 
# Optionality is commented to filter by lagged price, market cap, and exchange type, following the paper. 
# The unfiltered results seem to match well with the reported literature. 

import datetime as dt
from pathlib import Path

import altair as alt
import great_tables as gt
import polars as pl
import pandas as pd
import seaborn as sns
import sf_quant.data as sfd
import sf_quant.research as sfr
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
start = dt.date(1990, 1, 1)
end = dt.date(2009, 12, 31)
# start = dt.date(2012, 1, 1)
# end = dt.date(2024, 12, 31)
price_filter = 1
num_bins = 5
signal_name = "illiq"
results_folder = Path("results/experiment_2a")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

data = sfd.load_crsp_v2_daily(
    start=start,
    end=end,
    columns=[
        "date",
        "permno",
        "cusip",
        "ticker",
        "ret",
        "prc",
        "vol",
        "shrout",
        "primaryexch",
        "securitytype",
    ]
).with_columns(
    pl.col('shrout').mul('prc').alias('market_cap')
).rename({"ret": "return", "prc": "price", "vol": "daily_volume"}) #.filter(pl.col('primaryexch').eq("N") | pl.col('primaryexch').eq("A"))


# Compute signal
signals = data.sort("permno", "date").with_columns(
    pl.col('return').abs()
    .truediv(pl.col("daily_volume").mul(pl.col('price')))
    .rolling_mean(window_size=22, min_samples=22)
    # .ewm_mean(span=66, min_samples=66)
    # .truediv(pl.col('return').rolling_std(window_size=66, min_samples=66))
    .mul(10e6)
    .log1p()
    .shift(2)
    .over("permno")
    .alias(signal_name)
)


# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("permno").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col(signal_name).is_not_nan(),
    pl.col(signal_name).lt(np.inf),
    
)

signal_stats = sfr.get_signal_stats(filtered, column=signal_name)

signal_stats_table = (
    gt.GT(signal_stats)
    .tab_header(title=f"{signal_name} Summary Statistics")
    .cols_label(
        mean="Mean",
        std="Std.",
        min="Minimum",
        q25="25th Percentile",
        q50="50th Percentile",
        q75="75th Percentile",
        max="Maximum",
    )
    # .fmt_percent(["mean_return", "volatility"], decimals=2)
    .fmt_number(["mean", "std", "min", "max", "q25", "q50", "q75"], decimals=2)
    .opt_stylize(style=4, color="gray")
)

signal_stats_table_path = results_folder / f"{signal_name}_stats.png"
signal_stats_table.save(signal_stats_table_path, scale=3)

signal_values = filtered.select(signal_name).to_numpy().flatten()
distribution_chart_path = results_folder / f"{signal_name}_distribution.png"

plt.figure(figsize=(10, 6))
plt.hist(signal_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlim(0, 10)
plt.title(f"{signal_name} Distribution")
plt.xlabel(f"{signal_name} Value")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(distribution_chart_path)




# Create portfolios
labels = [str(i) for i in range(num_bins)]
portfolios = filtered.with_columns(
    pl.col(signal_name).qcut(num_bins, labels=labels).over("date").alias("bin")
)


# Target Volatility (we use 5% for Silver Fund)
target_vol = 0.05 

returns = (
    portfolios.group_by("date", "bin")
    .agg(pl.col("return").mean())

    .pivot(on="bin", index="date", values="return")
    .with_columns(
        (pl.col(str(num_bins - 1)) - pl.col("0")).alias("spread")
    )
    .unpivot(index="date", variable_name="bin", value_name="return")
    .sort("date", "bin")
    .with_columns(
        pl.col("return")
        .truediv(
            pl.col("return").std().over("bin")  
            * (pl.lit(252).sqrt() / target_vol) 
        )
    )
)

# Compute cumulative returns
cumulative_returns = returns.sort("date", "bin").select(
    "date",
    "bin",
    pl.col("return")
    .log1p()
    .cum_sum()
    .mul(100)
    .over("bin")
    .alias("cumulative_return"),
)

# Plot cumulative log returns
colors = sns.color_palette("coolwarm", num_bins).as_hex()
colors.append("black")
chart = (
    alt.Chart(cumulative_returns, title="Quantile Backtest Results")
    .mark_line()
    .encode(
        x=alt.X("date", title=""),
        y=alt.Y("cumulative_return", title="Cumulative Log Return (%)"),
        color=alt.Color(
            "bin",
            title="Portfolio",
            scale=alt.Scale(domain=["0", "1", "2", "3", "4", "spread"], range=colors),
        ),
    )
    .properties(width=800, height=400)
)

# Save chart
chart_path = results_folder / "cumulative_returns.png"
chart.save(chart_path, scale_factor=3)

# Create summary table
summary = (
    returns.group_by("bin")
    .agg(
        pl.col("return").mean().mul(252).alias("mean_return"),
        pl.col("return").std().mul(pl.lit(252).sqrt()).alias("volatility"),
    )
    .with_columns(pl.col("mean_return").truediv(pl.col("volatility")).alias("sharpe"))
    .sort("bin", descending=True)
)

table = (
    gt.GT(summary)
    .tab_header(title="Quantile Backtest Results")
    .cols_label(
        bin="Portfolio",
        mean_return="Mean Return",
        volatility="Volatility",
        sharpe="Sharpe",
    )
    .fmt_percent(["mean_return", "volatility"], decimals=2)
    .fmt_number("sharpe", decimals=2)
    .opt_stylize(style=4, color="gray")
)

table_path = results_folder / "summary_table.png"
table.save(table_path, scale=3)

# Fama french regression
ff5 = (
    sfd.load_fama_french(start=start, end=end)
    .sort("date")
    .with_columns(pl.exclude("date").shift(-1))
)

regression_data = (
    returns.join(other=ff5, on="date", how="left")
    .drop_nulls("return")
    .with_columns(pl.col("return").sub("rf").alias("return_rf"))
    .with_columns(pl.exclude("date", "bin").mul(100))
)

bins = labels + ["spread"]
regression_summary_list = []
for bin in bins:
    regression_data_slice = regression_data.filter(pl.col("bin").eq(bin))
    # formula = "return_rf ~ mkt_rf + smb + hml + rmw + cma"
    formula = "return_rf ~ mkt_rf + smb + hml"
    model = smf.ols(formula, regression_data_slice)
    results = model.fit()

    regression_summary = pl.DataFrame(
        {
            "variable": results.params.index,
            "coefficient": results.params.values,
            "tstat": results.tvalues.values,
        }
    ).with_columns(pl.lit(bin).alias("bin"))

    regression_summary_list.append(regression_summary)

regression_summary = (
    pl.concat(regression_summary_list)
    .sort("bin")
    .pivot(index="bin", on="variable", values=["coefficient", "tstat"])
)

regression_table = (
    gt.GT(regression_summary)
    .tab_header(title="Quantile Backtest Results (Daily %)")
    .cols_label(
        bin="Portfolio",
        coefficient_Intercept="Intercept Coef.",
        coefficient_mkt_rf="MKT Coef.",
        coefficient_smb="SMB Coef.",
        coefficient_hml="HML Coef.",
        # coefficient_rmw="RMW Coef.",
        # coefficient_cma="CMA Coef.",
        tstat_Intercept="Intercept T-stat",
        tstat_mkt_rf="MKT T-stat",
        tstat_smb="SMB T-stat",
        tstat_hml="HML T-stat",
        # tstat_rmw="RMW T-stat",
        # tstat_cma="CMA T-stat",
    )
    .fmt_number(pl.exclude("bin"), decimals=4)
    .opt_stylize(style=4, color="gray")
)

table_path = results_folder / "regression_table.png"
regression_table.save(table_path, scale=3)