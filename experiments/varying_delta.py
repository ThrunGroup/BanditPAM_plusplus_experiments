import numpy as np
import os
import pandas as pd
from scipy.spatial import distance_matrix

from run_all_versions import run_banditpam
from scaling_experiment import read_dataset
from scripts.comparison_utils import print_results, store_results
from scripts.constants import (
    SWAP_CONFIDENCE_ARR,
    MNIST,
    NEWSGROUPS,
    BANDITPAM_VA_CACHING,
    BANDITPAM_ORIGINAL_NO_CACHING,
    ALL_BANDITPAMS,
)


def run_delta_experiment(
    dataset_name,
    n_medoids_list=None,
    algorithms=None,
    loss: str = "L2",
    verbose=True,
    save_logs=True,
    cache_width=1000,
    dirname="varying_delta",
    num_experiments=3,
):
    """
    Runs the BanditPAM algorithm on varying delta to see how different confidence intervals affects
    the clustering. Stores resuls in appropriate log files.

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

    # TODO: make this more refined
    dataset = dataset[:10000]

    num_data = len(dataset)
    log_dir = os.path.join("logs", dirname)
    print("Running varying delta experiments on ", dataset_name)

    for experiment_index in range(num_experiments):
        print("\n\nExperiment: ", experiment_index)
        for n_medoids in n_medoids_list:
            print("\nNum medoids: ", n_medoids)
            for algorithm in algorithms:
                print("Running ", algorithm)

                # varying delta here
                for s_conf in [1]:
                    log_name = (
                        f"{algorithm}"
                        f"_{dataset_name}"
                        f"_n{num_data}"
                        f"_idx{experiment_index}"
                        f"_delta{s_conf}"   # TODO: this is swapConfidence --> delta is actually sqrt this with scaling
                    )
                    kmed, runtime = run_banditpam(
                        algorithm, dataset, n_medoids, loss, cache_width, swap_confidence=s_conf
                    )

                    if verbose:
                        print_results(kmed, runtime)

                    if save_logs:
                        store_results(
                            kmed, runtime, log_dir, log_name, num_data, n_medoids, confidence=s_conf
                        )


if __name__ == "__main__":
    run_delta_experiment(
        dataset_name=NEWSGROUPS,
        n_medoids_list=[10],
        algorithms=[BANDITPAM_VA_CACHING],
        loss="L2",
        verbose=True,
        save_logs=True,
        cache_width=50000,
        dirname="varying_delta_newsgroups_large",
        num_experiments=1,
    )