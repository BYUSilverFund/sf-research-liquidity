import datetime as dt
from pathlib import Path

import altair as alt
import great_tables as gt
import polars as pl
import seaborn as sns
import sf_quant.data as sfd
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

# print(sfd.get_crsp_v2_daily_columns())
print(sfd.get_exposures_columns())

# # Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
# price_filter = 5
# num_bins = 5
# signal_name = "illiq"
# results_folder = Path("results/test")

# # Create results folder
# results_folder.mkdir(parents=True, exist_ok=True)

data = sfd.load_exposures(
    start=start,
    end=end,
    in_universe=True,
    columns=[
        "date",
        "barrid",
        # "USSLOWL_LIQUIDTY",
        "USSLOWL_AIRLINES"
    ]
).drop_nulls()

print(data['USSLOWL_AIRLINES'].unique())


# print(data)