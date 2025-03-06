#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct ArgExpressionKwargs {
    k: i32,
}

fn impl_argpartition_i32(ca: &ChunkedArray<Int32Type>, k: usize) -> ChunkedArray<UInt32Type> {
    let mut indices: Vec<u32> = (0..ca.len() as u32).collect();

    let as_vec: Vec<i32> = ca.into_no_null_iter().collect();

    indices.select_nth_unstable_by_key(k, |&i| as_vec[i as usize]);

    UInt32Chunked::from_vec("idxs".into(), indices)
}

fn impl_argpartition_i64(ca: &ChunkedArray<Int64Type>, k: usize) -> ChunkedArray<UInt32Type> {
    let mut indices: Vec<u32> = (0..ca.len() as u32).collect();

    let as_vec: Vec<i64> = ca.into_no_null_iter().collect();

    indices.select_nth_unstable_by_key(k, |&i| as_vec[i as usize]);

    UInt32Chunked::from_vec("idxs".into(), indices)
}

fn impl_argpartition_f32(ca: &ChunkedArray<Float32Type>, k: usize) -> ChunkedArray<UInt32Type> {
    let mut indices: Vec<u32> = (0..ca.len() as u32).collect();

    indices.select_nth_unstable_by(k, |&i, &j| {
        ca.get(i as usize)
            .unwrap()
            .total_cmp(&ca.get(j as usize).unwrap())
    });

    UInt32Chunked::from_vec("idxs".into(), indices)
}

fn impl_argpartition_f64(ca: &ChunkedArray<Float64Type>, k: usize) -> ChunkedArray<UInt32Type> {
    let mut indices: Vec<u32> = (0..ca.len() as u32).collect();

    indices.select_nth_unstable_by(k, |&i, &j| {
        ca.get(i as usize)
            .unwrap()
            .total_cmp(&ca.get(j as usize).unwrap())
    });

    UInt32Chunked::from_vec("idxs".into(), indices)
}

#[polars_expr(output_type=UInt32)]
fn argpartition(inputs: &[Series], kwargs: ArgExpressionKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];

    if s.null_count() > 0 {
        polars_bail!(InvalidOperation: "argpartition does not support null values.")
    }

    let k = kwargs.k as usize;

    match s.dtype() {
        DataType::Int32 => Ok(impl_argpartition_i32(s.i32()?, k).into_series()),
        DataType::Int64 => Ok(impl_argpartition_i64(s.i64()?, k).into_series()),
        DataType::Float32 => Ok(impl_argpartition_f32(s.f32()?, k).into_series()),
        DataType::Float64 => Ok(impl_argpartition_f64(s.f64()?, k).into_series()),
        dtype => {
            polars_bail!(InvalidOperation:format!("dtype {dtype} not \
            supported for abs_numeric, expected Int32, Int64, Float32, Float64."))
        },
    }
}
