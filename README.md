# Polars argpartition

A [Polars plugin](https://docs.pola.rs/user-guide/plugins/expr_plugins/#writing-the-expression) that implements the [`argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html) function in Rust.

Under the hood, it uses the [`select_nth_unstable`](https://doc.rust-lang.org/std/primitive.slice.html#method.select_nth_unstable) method from the Rust standard library.

## Installation

```bash
pip install polars-argpartition
```

## Usage

```python
import polars as pl
from polars_argpartition import argpartition

df = pl.DataFrame(
    {
        "a": [1, 3, 6, 2, 5, 10, 12],
    }
)

print(df.with_columns(idxs=argpartition(pl.col("a"), k=3)))

```

Output (the order may change between runs, but the element at index 3 will always be the 4th smallest element):

```
shape: (7, 2)
┌─────┬───────┐
│ a   ┆ idxs  │
│ --- ┆ ---   │
│ i64 ┆ u32   │
╞═════╪═══════╡
│ 1   ┆ 0     │
│ 3   ┆ 3     │
│ 6   ┆ 1     │
│ 2   ┆ 4     │
│ 5   ┆ 2     │
│ 10  ┆ 5     │
│ 12  ┆ 6     │
└─────┴───────┘
```

Another use case of this function is to get top-k elements in a column without caring about their order. You can do that as follows 

```python
df = pl.DataFrame(
    {
        "a": [1, 3, 6, 2, 5, 10, 12],
    }
)

k = 3

print(
    df.with_columns(
        idxs=argpartition(
            -pl.col("a"),
            k=k,
        )
    )
    .with_row_index(name="row_index")
    .filter(pl.col("row_index").is_in(pl.col("idxs").slice(0, k)))
    .select(["a"])
)

```

Output:

```
shape: (3, 1)
┌──────┐
│ a    │
│ ---  │ 
│ i64  │
╞══════╡
│ 6    │
│ 10   │
│ 12   │ 
└──────┘
```

## Acknowledgments

A huge thank you to [Marco Gorelli](https://github.com/MarcoGorelli) for his fantastic [Polars plugin tutorial](https://marcogorelli.github.io/polars-plugins-tutorial/), which served as the foundation for this project.

