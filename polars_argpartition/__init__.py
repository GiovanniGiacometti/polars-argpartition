from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_argpartition._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars_argpartition.typing import IntoExprColumn

LIB = Path(__file__).parent


def argpartition(expr: IntoExprColumn, *, k: int) -> pl.Expr:
    """
    Returns the indices that would partition the given column
    such that the the element at index `k` is in
    its final sorted position, all elements before are less or equal
    and all elements after are greater or equal.
    Note that the order of elements before and after `k` is arbitrary
    and may change between runs.

    Parameters
    ----------
    expr
        The column to partition.
    k
        The index to partition by.

    Returns
    -------
    Expr
        Column of data type UInt32

    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="argpartition",
        is_elementwise=True,
        kwargs={"k": k},
    )
