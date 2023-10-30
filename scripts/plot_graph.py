from typing import List
import os
from matplotlib import pyplot as plt
import glob
import numpy as np
import pandas as pd

from constants import (
    MNIST,
    SCRNA,
    CIFAR,
    NEWSGROUPS,
    # algorithms
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_VA_CACHING,
    ALL_BANDITPAMS,
    # experiment settings
    NUM_DATA,
    VAR_DELTA,
    NUM_MEDOIDS,
    NUM_SWAPS,
    RUNTIME,
    SAMPLE_COMPLEXITY,
    LOSS,
    LOSS_HISTORY,
    # utils
    ALG_TO_COLOR,
    ALG_TO_LABEL,
)


# plt.rcParams["figure.figsize"] = (8, 6)


# from experiments.run_scaling_experiment import get_loss_function


def translate_experiment_setting(dataset, setting, num_seeds):
    """
    Translate a setting into a human-readable format For example, "k5" becomes
    "Num medoids: 5".
    TODO: removed seed in title
    """
    if dataset in [CIFAR, SCRNA]:
        loss = "L1"
    else:
        loss = "L2"

    if "k" in setting:
        return (
            f"({dataset}, ${setting[0]}={setting[1:]}$, {loss})"
            # , Seeds ="
            # f" {num_seeds})"
        )
    elif "n" in setting:
        return (
            f"({dataset}, ${setting[0]}={setting[1:]}$)"
            # f", Seeds "
            # f"= {num_seeds})"
        )
    else:
        assert False, "Invalid setting"


def get_x_label(x_axis, is_logspace_x):
    if x_axis == NUM_DATA:
        x_label = "Dataset size ($n$)"
    elif x_axis == VAR_DELTA:
        x_label = "Delta"
    elif x_axis == NUM_MEDOIDS:
        x_label = "Number of medoids ($k$)"
    elif x_axis == NUM_SWAPS:
        x_label = "Number of swaps ($T$)"
    else:
        raise Exception("Bad x label")

    if is_logspace_x:
        x_label = f"ln({x_label})"

    return x_label


def get_x(data, x_axis, is_logspace_x):
    if x_axis == NUM_SWAPS:
        x = list(range(10))
    else:
        x = data[x_axis].tolist()

    if is_logspace_x:
        x = np.log(x)

    return x


def get_y_label(y_axis, is_logspace_y):
    if y_axis is LOSS:
        y_label = "Final Loss Normalized to BanditPAM ($L/L_{BanditPAM}$)"
    elif y_axis is LOSS_HISTORY:
        y_label = "Loss"
    elif y_axis is SAMPLE_COMPLEXITY:
        y_label = "Sample Complexity"
    else:
        y_label = "Wall-Clock Runtime"

    if is_logspace_y:
        y_label = f"ln({y_label})"

    return y_label


def get_y_and_error(
    y_axis,
    data_mean,
    data_std,
    algorithm,
    is_logspace_y,
    num_experiments,
    baseline_losses=1.0,
    bpam_loss_history=None,
):
    if y_axis is LOSS:
        # Plot the loss divided by that of
        # Original BanditPam without Caching
        y = np.array(data_mean[y_axis].tolist())
        if algorithm == BANDITPAM_ORIGINAL_NO_CACHING:
            # The first algorithm is
            # BANDITPAM_ORIGINAL_NO_CACHING
            baseline_losses = y.copy()
        y /= baseline_losses
        error = data_std[y_axis].tolist()
    elif y_axis is LOSS_HISTORY:
        y = data_mean["loss"].tolist()
        error = data_std["loss"].tolist()
        if algorithm == BANDITPAM_ORIGINAL_NO_CACHING:
            bpam_loss_history = y
        else:
            y = bpam_loss_history
    elif y_axis is SAMPLE_COMPLEXITY:
        y = data_mean["average_complexity_with_caching"].tolist()
        error = data_std["average_complexity_with_caching"].tolist()
    else:
        y = np.array(data_mean["average_runtime"].tolist())
        error = data_std["average_runtime"].tolist()
        # y = data_mean["total_runtime"].tolist()
        # error = data_std["total_runtime"].tolist()

    if is_logspace_y:
        y = np.log(y)
        error = np.log(error) / y / np.sqrt(num_experiments)

    return y, error, baseline_losses


def get_titles(x_axis, y_axis, y_label, dataset, setting, num_seeds):
    if y_axis == LOSS:
        y_title = "$L/L_{BanditPAM}$"
    elif y_axis == LOSS_HISTORY:
        y_title = "Loss"
    else:
        y_title = y_label
    if x_axis == NUM_DATA:
        x_title = "$n$"
    elif x_axis == NUM_SWAPS:
        x_title = "$T$"
    else:
        x_title = "$k$"
    title = (
        f"{y_title} vs. {x_title} "
        f"{translate_experiment_setting(dataset, setting, num_seeds)}"
    )
    return x_title, y_title, title


def create_scaling_plots(
    datasets: List[str] = [],
    algorithms: List[str] = [],
    x_axes=NUM_DATA,
    y_axes=RUNTIME,
    is_logspace_x: bool = False,
    is_logspace_y: bool = False,
    include_error_bar: bool = False,
    dir_name: str = None,
    settings: List[str] = None,
):
    """
    Plot the scaling experiments from the data stored in the logs file.

    :param datasets: the datasets that you want to plot. If empty, warning is
        triggered.
    :param algorithms: the algorithms that you want to plot. If empty, warning
        is triggered.
    :param include_error_bar: shows the standard deviation
    :param is_logspace_x: whether to plot x-axis in logspace or not
    :param is_logspace_y: whether to plot y-axis in logspace or not
    :param dir_name: directory name of log files
    """
    if len(datasets) == 0:
        raise Exception("At least one dataset must be specified")

    if len(algorithms) == 0:
        raise Exception("At least one algorithm must be specified")

    for x_axis in x_axes:
        # get log csv files
        if dir_name is None:
            parent_dir = os.path.dirname(os.path.abspath(__file__))

            if x_axis == VAR_DELTA:
                log_dir_name = "varying_delta"
            elif x_axis == NUM_DATA:
                log_dir_name = "scaling_with_n_cluster"
            else:
                log_dir_name = "scaling_with_k_cluster"

            log_dir = os.path.join(parent_dir, "logs", log_dir_name)
        else:
            root_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            log_dir = os.path.join(root_dir, "logs", dir_name)

        for dataset in datasets:
            csv_files = glob.glob(os.path.join(log_dir, f"*{dataset}*"))

            # list available settings by parsing the log file names.
            # for example, "settings" returns ["k5", "k10"]
            # if the experiment used "k = 5" and "k = 10" settings
            if not settings:
                settings = list(
                    set([file.split("_")[-2] for file in csv_files])
                )

            for setting in settings:
                for y_axis in y_axes:
                    baseline_losses = 1.0
                    bpam_loss_history = []
                    for algorithm in algorithms:
                        algorithm_files = glob.glob(
                            os.path.join(
                                log_dir,
                                f"*{algorithm}*{dataset}*{setting}*idx*",
                            )
                        )
                        num_seeds = len(algorithm_files)
                        algorithm_dfs = [
                            pd.read_csv(file) for file in algorithm_files
                        ]
                        data = pd.concat(algorithm_dfs)
                        data_mean = data.groupby(data.index).mean()
                        data_std = data.groupby(data.index).std() / np.sqrt(
                            len(data)
                        )

                        # Set x axis
                        x_label = get_x_label(x_axis, is_logspace_x)
                        x = get_x(data, x_axis, is_logspace_x)

                        # Set y axis
                        y_label = get_y_label(y_axis, is_logspace_y)
                        num_experiments = len(algorithm_files)
                        (y, error, baseline_losses) = get_y_and_error(
                            y_axis,
                            data_mean,
                            data_std,
                            algorithm,
                            is_logspace_y,
                            num_experiments,
                            baseline_losses,
                            bpam_loss_history,
                        )

                        if algorithm == BANDITPAM_ORIGINAL_NO_CACHING:
                            bpam_loss_history = y

                        # Sort the (x, y) pairs by the ascending order of x
                        x, y = zip(
                            *sorted(zip(x, y), key=lambda pair: pair[0])
                        )

                        plt.scatter(
                            x,
                            y,
                            color=ALG_TO_COLOR[algorithm],
                            label=ALG_TO_LABEL[algorithm],
                        )
                        plt.plot(x, y, color=ALG_TO_COLOR[algorithm])

                        if include_error_bar:
                            plt.errorbar(
                                x,
                                y,
                                yerr=np.abs(error),
                                fmt=".",
                                color="black",
                            )

                        # Sort the legend entries (labels and handles)
                        # by labels
                        handles, labels = plt.gca().get_legend_handles_labels()
                        plt.legend(handles, labels, loc="upper left")

                        x_title, y_title, title = get_titles(
                            x_axis,
                            y_axis,
                            y_label,
                            dataset,
                            setting,
                            num_seeds,
                        )
                        plt.title(title)
                        plt.xlabel(x_label)
                        plt.ylabel(y_label)

                    plt.show()


if __name__ == "__main__":
    # This is for scaling with k on newsgroups
    create_scaling_plots(
        datasets=[NEWSGROUPS],
        algorithms=ALL_BANDITPAMS,
        x_axes=[NUM_MEDOIDS],
        y_axes=[SAMPLE_COMPLEXITY, RUNTIME],
        is_logspace_y=False,
        dir_name="scaling_with_k",
        include_error_bar=True,
    )

    # create_scaling_plots(
    #     datasets=[MNIST, CIFAR],
    #     algorithms=[BANDITPAM_ORIGINAL_NO_CACHING, BANDITPAM_VA_CACHING],
    #     x_axes=[NUM_SWAPS],
    #     y_axes=[LOSS_HISTORY],
    #     is_logspace_y=False,
    #     dir_name="swap_vs_loss_2",
    #     include_error_bar=False,
    #     settings=["k5", "k10"],
    # )

    # create_scaling_plots(
    #     datasets=[MNIST],
    #     algorithms=[BANDITPAM_ORIGINAL_NO_CACHING, BANDITPAM_VA_CACHING],
    #     x_axes=[NUM_DATA],
    #     y_axes=[RUNTIME],
    #     is_logspace_y=False,
    #     dir_name="mnist",
    #     include_error_bar=True,
    # )
