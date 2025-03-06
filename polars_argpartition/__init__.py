from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_argpartition._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars_argpartition.typing import IntoExprColumn

LIB = Path(__file__).parent


def arg_partition(expr: IntoExprColumn, *, k: int) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="arg_partition",
        is_elementwise=True,
        kwargs={"k": k},
    )
