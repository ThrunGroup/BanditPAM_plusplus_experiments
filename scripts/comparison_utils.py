import os
import pandas as pd


def print_results(kmed, runtime):
    print("-----Results-----")
    print("Algorithm:", kmed.algorithm)
    print("Final Medoids:", kmed.medoids)
    print("Loss:", kmed.average_loss)
    print("Misc complexity:", f"{kmed.misc_distance_computations:,}")
    print("Build complexity:", f"{kmed.build_distance_computations:,}")
    print("Swap complexity:", f"{kmed.swap_distance_computations:,}")
    print("Number of Swaps", kmed.steps)
    print("Cache Writes: {:,}".format(kmed.cache_writes))
    print("Cache Hits: {:,}".format(kmed.cache_hits))
    print("Cache Misses: {:,}".format(kmed.cache_misses))
    print(
        "Average Swap Sample Complexity:",
        f"{kmed.swap_distance_computations / kmed.steps:,}",
    )
    print(
        "Total complexity (without misc):",
        f"{kmed.getDistanceComputations(False):,}"
    )
    print(
        "Total complexity (with misc):",
        f"{kmed.getDistanceComputations(True):,}",
    )
    print("Runtime per swap:", runtime / kmed.steps)
    print("Total runtime:", runtime)


def store_results(kmed, runtime, log_dir, log_name, num_data, num_medoids):
    # Create a dictionary with the printed values
    log_dict = {
        "num_data": num_data,
        "num_medoids": num_medoids,
        "loss": kmed.average_loss,
        "misc_complexity": kmed.misc_distance_computations,
        "build_complexity": kmed.build_distance_computations,
        "swap_complexity": kmed.swap_distance_computations,
        "number_of_swaps": kmed.steps,
        "cache_writes": kmed.cache_writes,
        "cache_hits": kmed.cache_hits,
        "cache_misses": kmed.cache_misses,
        "average_swap_sample_complexity": kmed.swap_distance_computations / kmed.steps,
        "total_complexity_without_misc": kmed.getDistanceComputations(False),
        "total_complexity_with_misc": kmed.getDistanceComputations(True),
        "runtime_per_swap": runtime / kmed.steps,
        "total_runtime": runtime
    }
    log_pd_row = pd.DataFrame([log_dict])

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}.csv")
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
    else:
        log_df = pd.DataFrame(columns=log_dict.keys())

    # Find the matching row to take a mean
    mask = (log_df["num_data"] == num_data) & (log_df["num_medoids"] == num_medoids)
    matching_row = log_df.loc[mask]

    if matching_row.empty:
        # Append the dictionary to the dataframe
        log_df = pd.concat([log_df, log_pd_row], ignore_index=True)
    else:
        mean_row = pd.concat([log_pd_row, matching_row], ignore_index=True).mean(axis=0)
        # Convert mean_row to a DataFrame with the same index as matching_row
        mean_df = pd.DataFrame([mean_row.values], columns=log_df.columns, index=matching_row.index)
        log_df.update(mean_df)  # Update log_df using mean_df

    # Save the updated dataframe back to the CSV file
    log_df.to_csv(log_path, index=False)
    print("Saved log to ", log_path)

