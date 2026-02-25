import numpy as np
import polars as pl


def liquidity() -> pl.Expr:
    return np.abs(pl.col("return")).truediv(pl.col("daily_volume").mul(pl.col("price")))
