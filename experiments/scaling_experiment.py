import numpy as np
import os
import pandas as pd
from scipy.spatial import distance_matrix

from run_all_versions import run_algorithm
from scripts.comparison_utils import print_results, store_results
from data.newsgroups_to_csv import twenty_newsgroup_to_csv
from scripts.constants import (
    # Datasets
    MNIST,
    CIFAR,
    NEWSGROUPS,
    # Algorithms
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_VA_CACHING,
)


def read_dataset(dataset_name):
    """
    Reads the specified dataset from the local storage.

    :param dataset_name: A string that represents the name of the dataset
    :return: The requested dataset as a numpy array
    """
    if dataset_name == MNIST:
        filename = "MNIST_70k"
        delimiter = " "
    elif dataset_name == CIFAR:
        filename = "cifar10"
        delimiter = ","
    elif dataset_name == NEWSGROUPS:
        filename = "20_newsgroups"
        delimiter = ","
        if not os.path.exists(os.path.join("data", f"{filename}.csv")):
            print("processing newsgroups dataset")
            twenty_newsgroup_to_csv()
    else:
        filename = "reduced_scrna"
        delimiter = ","

    dataset = pd.read_csv(
        os.path.join("data", f"{filename}.csv"),
        delimiter=delimiter,
        header=None,
    ).to_numpy()

    if dataset_name == NEWSGROUPS:
        dataset = dataset[1:, 1:]

    print(dataset.shape)
    return dataset


def scaling_experiment_with_k(
    dataset_name,
    n_medoids_list=None,
    num_data=10000,
    algorithms=None,
    loss: str = "L2",
    verbose=True,
    save_logs=True,
    cache_width=1000,
    dirname="scaling_with_k",
    num_experiments=3,
    parallelize=True,
    n_swaps=10,
    build_confidence=3,
    swap_confidence=10,
):
    """
    Runs a scaling experiment varying the number of medoids (k), and stores the
    results in the appropriate log files.

    :param dataset_name: A string that represents the name of the dataset
    :param n_medoids_list: A list of integers specifying different number of
        medoids to run the experiment with
    :param algorithms: A list of strings specifying the names of algorithms to
        use in the experiment
    :param loss: A string specifying the type of loss to be used
    :param verbose: A boolean indicating whether to print the results
    :param save_logs: A boolean indicating whether to save the results
    :param cache_width: An integer specifying the cache width for BanditPAM
    :param dirname: A string directory name where the log files will be saved
    :param num_experiments: The number of experiments to run
    """
    dataset = read_dataset(dataset_name)
    log_dir = os.path.join("logs", dirname)
    print("Running sampling complexity experiment with k on ", dataset_name)

    for experiment_index in range(num_experiments):
        print("\n\nExperiment: ", experiment_index)

        for n_medoids in n_medoids_list:
            print("\nNum medoids: ", n_medoids)
            data_indices = np.random.randint(0, len(dataset), num_data)
            dataset = dataset[data_indices]

            for algorithm in algorithms:
                print("Running ", algorithm)
                log_name = (
                    f"{algorithm}"
                    f"_{dataset_name}"
                    f"_n{num_data}"
                    f"_idx{experiment_index}"
                )

                if "with" in algorithm:
                    algorithm_name = BANDITPAM_VA_CACHING
                else:
                    algorithm_name = BANDITPAM_ORIGINAL_NO_CACHING

                build_only_csv = pd.read_csv(
                    os.path.join(
                        "logs",
                        "build_only_all",
                        f"{algorithm_name}_{dataset_name}_k{n_medoids}_idx0.csv",  # TODO(@motiwari): should this be {experiment_index}?
                    )
                )
                build_only_time = build_only_csv["total_runtime"][0]

                kmed, runtime = run_algorithm(
                    algorithm,
                    dataset,
                    n_medoids,
                    loss,
                    n_swaps=n_swaps,
                    cache_width=cache_width,
                    parallelize=parallelize,
                    build_confidence=build_confidence,
                    swap_confidence=swap_confidence,
                )

                if verbose:
                    print_results(kmed, runtime, build_only_time)

                if save_logs:
                    store_results(
                        kmed,
                        runtime,
                        log_dir,
                        log_name,
                        num_data,
                        n_medoids,
                        save_loss_history=False,
                        build_only_time=build_only_time,
                    )

def scaling_experiment_with_n(
    dataset_name,
    num_data_list,
    n_medoids,
    algorithms=None,
    loss: str = "L2",
    verbose=True,
    save_logs=True,
    cache_width=20000,
    dirname="mnist",
    parallelize=True,
    num_experiments=3,
    n_swaps=10,
    build_confidence=3,
    swap_confidence=10,
    save_loss_history=True,
    num_data_indices=[0, 1, 2, 3],
):
    """
    Runs a scaling experiment varying the number of data points (n), and stores
    the results in the appropriate log files.

    :param dataset_name: A string that represents the name of the dataset
    :param num_data_list: A list of integers specifying different number of
        data points to run the experiment with
    :param n_medoids: An integer specifying the number of medoids
    :param algorithms: A list of strings specifying the names of algorithms to
        use in the experiment
    :param loss: A string specifying the type of loss to be used
    :param verbose: A boolean indicating whether to print the results
    :param save_logs: A boolean indicating whether to save the results
    :param cache_width: An integer specifying the cache width for BanditPAM
    :param dirname: A string specifying the directory name where the log files
        will be saved
    :param num_experiments: An integer specifying the number of experiments to
        run
    """
    dataset = read_dataset(dataset_name)
    log_dir = os.path.join("logs", dirname)

    print("Running sampling complexity experiment with n on ", dataset_name)

    for experiment_index in range(num_experiments):
        print("\n\nExperiment: ", experiment_index)
        for num_data_index in num_data_indices:
            num_data = num_data_list[num_data_index]
            print("\nNum data: ", num_data)
            data_indices = np.random.randint(0, len(dataset), num_data)
            data = dataset[data_indices]
            for algorithm in algorithms:
                print("\n<Running ", algorithm, ">")
                log_name = (
                    f"{algorithm}"
                    f"_{dataset_name}"
                    f"_k{n_medoids}"
                    f"_idx{experiment_index}"
                )
                print(log_name)

                if n_swaps == 0:
                    build_only_time = 0
                else:
                    if "with" in algorithm:
                        algorithm_name = BANDITPAM_VA_CACHING
                    else:
                        algorithm_name = BANDITPAM_ORIGINAL_NO_CACHING
                    build_only_csv = pd.read_csv(
                        os.path.join(
                            "logs",
                            "build_only_all",
                            f"{algorithm_name}_{dataset_name}_k{n_medoids}_idx0.csv",
                        )
                    )
                    build_only_time = build_only_csv["total_runtime"][
                        num_data_index
                    ]

                kmed, runtime = run_algorithm(
                    algorithm,
                    data,
                    n_medoids,
                    loss,
                    cache_width,
                    parallelize,
                    n_swaps=n_swaps,
                    build_confidence=build_confidence,
                    swap_confidence=swap_confidence,
                )

                if verbose:
                    print_results(kmed, runtime, build_only_time)

                if save_logs:
                    store_results(
                        kmed,
                        runtime,
                        log_dir,
                        log_name,
                        num_data,
                        n_medoids,
                        save_loss_history,
                        build_only_time,
                    )


def debug(
    dataset_name,
    num_data_list,
    n_medoids,
    algorithms=None,
    loss: str = "L2",
    verbose=True,
    save_logs=True,
    cache_width=1000,
    dirname="mnist",
    parallelize=True,
    num_experiments=3,
    num_swaps=10,
    build_confidence=3,
    swap_confidence=10,
):
    """
    Runs a scaling experiment varying the number of data points (n), and stores
    the results in the appropriate log files.

    :param dataset_name: A string that represents the name of the dataset
    :param num_data_list: A list of integers specifying different number of
        data points to run the experiment with
    :param n_medoids: An integer specifying the number of medoids
    :param algorithms: A list of strings specifying the names of algorithms to
        use in the experiment
    :param loss: A string specifying the type of loss to be used
    :param verbose: A boolean indicating whether to print the results
    :param save_logs: A boolean indicating whether to save the results
    :param cache_width: An integer specifying the cache width for BanditPAM
    :param dirname: A string specifying the directory name where the log files
        will be saved
    :param num_experiments: An integer specifying the number of experiments to
        run
    """
    dataset = read_dataset(dataset_name)
    log_dir = os.path.join("logs", dirname)

    print("Running sampling complexity experiment with n on ", dataset_name)

    for experiment_index in range(8, num_experiments):
        print("\n\nExperiment: ", experiment_index)
        for num_data in num_data_list:
            print("\nNum data: ", num_data)
            data_indices = np.random.randint(0, len(dataset), num_data)
            data = dataset[data_indices]
            for algorithm in algorithms:
                print("\n<Running ", algorithm, ">")
                log_name = (
                    f"{algorithm}"
                    f"_{dataset_name}"
                    f"_k{n_medoids}"
                    f"_idx{experiment_index}"
                )
                print(log_name)
                kmed, runtime = run_algorithm(
                    algorithm,
                    data,
                    n_medoids,
                    loss,
                    cache_width,
                    parallelize,
                    n_swaps=num_swaps,
                    build_confidence=build_confidence,
                    swap_confidence=swap_confidence,
                )

                banditpam_build_medoids = data[kmed.medoids, :]

                banditpam_medoids_ref_cost_distance_matrix = distance_matrix(
                    banditpam_build_medoids,
                    data
                )
                banditpam_objective = np.sum(
                    np.min(banditpam_medoids_ref_cost_distance_matrix, 0)
                )

                print("loss: ", banditpam_objective)

                if verbose:
                    print_results(kmed, runtime)

                if save_logs:
                    store_results(
                        kmed, runtime, log_dir, log_name, num_data, n_medoids
                    )