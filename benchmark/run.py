from collections import defaultdict
import os
import numpy as np
import polars as pl
import polars_argpartition as pl_ap
import random
import time


n_runs = 10

column_data_types = [pl.Int64, pl.Float64, pl.Int32, pl.Float32]

for column_data_type in column_data_types:
    final_results = defaultdict(list)
    for len_df in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        curr_results = defaultdict(list)
        for _ in range(n_runs):
            df = pl.DataFrame(
                {"a": random.choices(range(len_df), k=len_df)}, schema={"a": pl.Int64}
            )

            print("Current shape of DataFrame:", df.shape)

            # print("Top 5 rows")
            # print(df.top_k(by="a", k=5))

            # print("Bottom 5 rows")
            # print(df.bottom_k(by="a", k=5))

            # print("Executing Polars argpartition")
            start_ap = time.time()
            idxs = df.with_columns(idxs=pl_ap.argpartition(pl.col("a"), k=5))
            # print(idxs)
            total_time_ap = time.time() - start_ap
            # print(f"Polars argpartition took {time.time() - start_ap:.2f} seconds")

            del idxs
            # print("Gathering the result")
            # # take first n rows
            # print(
            #     idxs.with_row_index(name="row_index").filter(
            #         pl.col("row_index").is_in(pl.col("idxs").slice(0, 5))
            #     )
            # )

            # print("Executing Polars sort")
            start_sort = time.time()
            idxs = df.sort("a")
            # print(idxs)
            total_time_sort = time.time() - start_sort
            # print(f"Polars sort took {time.time() - start:.2f} seconds")

            curr_results["argpartition"].append(total_time_ap)
            curr_results["sort"].append(total_time_sort)

            del df
            del idxs

        final_results["len_df"].append(len_df)
        final_results["argpartition (average in s)"].append(
            sum(curr_results["argpartition"]) / n_runs
        )
        final_results["sort (average in s)"].append(sum(curr_results["sort"]) / n_runs)
        final_results["argpartition (std in s)"].append(
            np.std(curr_results["argpartition"])
        )
        final_results["sort (std in s)"].append(np.std(curr_results["sort"]))

    results = pl.DataFrame(final_results)

    print("Results for data type:", column_data_type)
    print(results)
    results.write_csv(
        os.path.join("benchmark", "results", f"results_{column_data_type}.csv")
    )
