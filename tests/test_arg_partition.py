import polars as pl
import pytest
import polars_argpartition as pl_ap
import random
from polars.testing import assert_frame_equal


@pytest.mark.parametrize("data_type", [pl.Int64, pl.Float64, pl.Int32, pl.Float32])
@pytest.mark.parametrize("n_rows", [1_000, 10_000, 100_000])
def test_argpartition(data_type, n_rows):
    """
    Test argpartition parametrizing over data types and number of rows.
    """
    df = pl.DataFrame(
        {"a": random.choices(range(n_rows, n_rows + n_rows), k=n_rows)},
        schema={"a": data_type},
    )

    top_5 = df.top_k(by="a", k=5)
    top_5 = top_5.sort("a")

    k = len(df) - 5

    idxs = df.with_columns(idxs=pl_ap.argpartition(pl.col("a"), k=k))

    df_gathered = (
        idxs.with_row_index(name="row_index")
        .filter(pl.col("row_index").is_in(pl.col("idxs").slice(k, len(df))))
        .sort("a")
        .select(["a"])
    )

    print("Top 5 rows")
    print(top_5)

    assert_frame_equal(top_5, df_gathered)
