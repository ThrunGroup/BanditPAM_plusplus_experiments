import os
import pandas as pd


def print_results(kmed, runtime, build_only_time):
    complexity_with_caching = (
        kmed.getDistanceComputations(True) - kmed.cache_hits
    )
    loss_history = kmed.losses
    interpolated_loss_history = loss_history + [loss_history[-1]] * (
        10 - len(loss_history)
    )
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
        "Total complexity (without misc):",
        f"{kmed.getDistanceComputations(False):,}",
    )
    print(
        "Total complexity (with misc):",
        f"{kmed.getDistanceComputations(True):,}",
    )
    print(
        "Total complexity (with caching):",
        f"{complexity_with_caching:,}",
    )
    print(
        "average_complexity_with_caching: ",
        complexity_with_caching / (kmed.steps + 1),
    )
    print("Runtime per swap:", runtime / (kmed.steps + 1))
    print("Swap ONLY: ", (runtime - build_only_time))

    if kmed.steps != 0:
        print(
            "average_swap_runtime: ", (runtime - build_only_time) / kmed.steps
        )

    print("Total runtime:", runtime)
    print("Losses: ", interpolated_loss_history)

def get_loss_history_log_dict(kmed, num_data, num_medoids):
    # Create a dictionary with the printed values
    loss_history = kmed.losses
    interpolated_loss_history = loss_history + [loss_history[-1]] * (
        10 - len(loss_history)
    )

    log_dicts = [
        {
            "num_swaps": i,
            "num_data": num_data,
            "num_medoids": num_medoids,
            "loss": interpolated_loss_history[i],
        }
        for i in range(10)
    ]

    return log_dicts


def store_results(
    kmed,
    runtime,
    log_dir,
    log_name,
    num_data,
    num_medoids,
    save_loss_history=True,
    build_only_time=0,
):
    # Create a dictionary with the printed values
    loss_history = kmed.losses
    interpolated_loss_history = loss_history + [loss_history[-1]] * (
        10 - len(loss_history)
    )

    log_dict = [
        {
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
            # "average_swap_sample_complexity": kmed.swap_distance_computations / kmed.steps,
            "total_complexity_without_misc": kmed.getDistanceComputations(
                False
            ),
            "total_complexity_with_misc": kmed.getDistanceComputations(True),
            "total_complexity_with_caching": kmed.getDistanceComputations(True)
            - kmed.cache_hits,
            "average_complexity_with_caching": (
                kmed.getDistanceComputations(True) - kmed.cache_hits
            )
            / (kmed.steps + 1),
            # "runtime_per_swap": runtime / kmed.steps,
            "average_runtime": runtime / (kmed.steps + 1),
            "total_swap_runtime": (runtime - build_only_time),
            "average_swap_runtime": 0
            if kmed.steps == 0
            else (runtime - build_only_time) / kmed.steps,
            "total_runtime": runtime,
            # "loss_history": interpolated_loss_history
            # if save_loss_history
            # else "",
        }
    ]

    if save_loss_history:
        log_dict = get_loss_history_log_dict(kmed, num_data, num_medoids)

    log_pd_row = pd.DataFrame(log_dict)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}.csv")
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
    else:
        log_df = pd.DataFrame(columns=log_dict[0].keys())

    # Append the dictionary to the dataframe
    log_df = pd.concat([log_df, log_pd_row], ignore_index=True)

    # Save the updated dataframe back to the CSV file
    log_df.to_csv(log_path, index=False)
    print("Saved log to ", log_path)
