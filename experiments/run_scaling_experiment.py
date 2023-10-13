import numpy as np

from experiments.scaling_experiment import (
    scaling_experiment_with_k,
    scaling_experiment_with_n,
    debug,
)
from scripts.constants import (
    # Algorithms
    ALL_BANDITPAMS,
    BANDITPAM_VA_NO_CACHING,
    # Datasets
    MNIST,
    CIFAR,
    SCRNA,
    NEWSGROUPS,
    # Parameters
    K_LIST,
)


def get_loss_function(dataset):
    """
    Returns the appropriate loss function based on the dataset.

    :param dataset: A string that represents the name of the dataset
    :return: A string indicating the type of loss function 
    """
    if dataset in [NEWSGROUPS]:
        return "cos"
    elif dataset in [MNIST]:
        return "L2"
    elif dataset in [CIFAR, SCRNA]:
        return "L1"
    else:
        raise Exception("Bad dataset name")


def get_num_data_list(dataset):
    """
    Returns a list of numbers indicating the different number of data points to
    run the experiment with, based on the dataset.

    :param dataset: A string that represents the name of the dataset
    :return: A numpy array specifying different numbers of data points
    """
    if dataset == MNIST:
        num_data = 70000
    elif dataset == CIFAR:
        num_data = 50000
    elif dataset == NEWSGROUPS:
        num_data = 50000
    elif dataset == SCRNA:
        num_data = 40000
    else:
        raise Exception("Bad dataset name")

    return np.linspace(10000, num_data, 4, dtype=int)


def run_scaling_experiment_with_k():
    """
    Runs scaling experiments varying the number of medoids k for all datasets using all BanditPAM algorithms.
    """

    for dataset in [MNIST, CIFAR, SCRNA, NEWSGROUPS]:
        parallelize = dataset != SCRNA
        loss = get_loss_function(dataset)
        scaling_experiment_with_k(
            dataset_name=dataset,
            loss=loss,
            algorithms=ALL_BANDITPAMS,
            n_medoids_list=K_LIST,
            cache_width=40000,
            parallelize=parallelize,
            n_swaps=3,
        )


def run_build_only():
    """
    Runs scaling experiments varying the number of data points (n) for the
    provided datasets using all BanditPAM algorithms.
    """
    for dataset in [MNIST, CIFAR, SCRNA]:
        loss = get_loss_function(dataset)
        num_data_list = get_num_data_list(dataset)
        parallelize = dataset != SCRNA

        if dataset == MNIST:
            swap_confidence = 30
        elif dataset == CIFAR:
            swap_confidence = 30
        elif dataset == SCRNA:
            swap_confidence = 15

        for n_medoids in K_LIST:
            np.random.seed(0)
            scaling_experiment_with_n(
                dataset_name=dataset,
                loss=loss,
                algorithms=ALL_BANDITPAMS,
                n_medoids=n_medoids,
                num_data_list=num_data_list,
                dirname="build_only",
                num_experiments=3,
                parallelize=parallelize,
                n_swaps=0,
                save_loss_history=False,
                cache_width=num_data_list[-1],
                swap_confidence=swap_confidence,
            )


def run_speedup_summary_table():
    """
    Runs scaling experiments varying the number of data points (n) for the
    MNIST and CIFAR datasets using all BanditPAM algorithms.
    """
    for dataset in [
        MNIST,
        CIFAR,
        SCRNA,
        NEWSGROUPS,
    ]:
        loss = get_loss_function(dataset)
        num_data_list = get_num_data_list(dataset)
        parallelize = dataset != SCRNA

        if dataset == MNIST:
            swap_confidence = 30
        elif dataset == CIFAR:
            swap_confidence = 30
        elif dataset == SCRNA:
            swap_confidence = 30

        for n_medoids in K_LIST:
            for algorithm in ALL_BANDITPAMS:
                np.random.seed(1)
                scaling_experiment_with_n(
                    dataset_name=dataset,
                    loss=loss,
                    algorithms=[algorithm],
                    n_medoids=n_medoids,
                    num_data_list=num_data_list,
                    dirname="speedup_summary",
                    num_experiments=3,
                    parallelize=parallelize,
                    n_swaps=5,
                    save_loss_history=False,
                    cache_width=num_data_list[-1],
                    build_confidence=3,
                    swap_confidence=swap_confidence,
                )


def run_swap_vs_loss():
    """
    Runs scaling experiments varying the number of data points (n) for the
    MNIST and CIFAR datasets using all BanditPAM algorithms.

    TODO: If the algorithms use the num swaps less than 10, just interpolate them and take an average
    """
    for dataset in [MNIST, CIFAR]:
        loss = get_loss_function(dataset)
        num_data_list = get_num_data_list(dataset)
        for n_medoids in [5, 10]:
            np.random.seed(0)
            scaling_experiment_with_n(
                dataset_name=dataset,
                loss=loss,
                algorithms=ALL_BANDITPAMS,
                n_medoids=n_medoids,
                num_data_list=num_data_list,
                dirname="swap_vs_loss",
                num_experiments=3,
                parallelize=False,
                n_swaps=5,
                build_confidence=3,
                swap_confidence=5,
                save_loss_history=True,
            )

def run_debug_scrna():
    """
    Used to debug the weird results we got on scRNA dataset when using parallelization.

    """
    for dataset in [SCRNA]:
        loss = get_loss_function(dataset)
        num_data_list = [3000]
        for par in [False]:
            print("\n\n\n\nParallelize: ", par)
            num_iters = 5 if par else 2

            for _ in range(num_iters):
                np.random.seed(0)
                debug(
                    dataset_name=dataset,
                    loss=loss,
                    algorithms=[BANDITPAM_VA_NO_CACHING],
                    n_medoids=5,
                    num_data_list=num_data_list,
                    dirname="new_scrna",
                    parallelize=par,
                    num_experiments=1,
                    num_swaps=5,
                    save_logs=False,
                )


if __name__ == "__main__":
    run_scaling_experiment_with_k()
    run_build_only()
    run_swap_vs_loss()
    run_speedup_summary_table()