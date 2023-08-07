import numpy as np

from scaling_experiment import (
    scaling_experiment_with_k,
    scaling_experiment_with_n,
    debug,
)
from scripts.constants import (
    # Algorithms
    ALL_BANDITPAMS,
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_ORIGINAL_CACHING,
    BANDITPAM_VA_NO_CACHING,
    BANDITPAM_VA_CACHING,
    # Datasets
    MNIST,
    CIFAR,
    SCRNA,
    NEWSGROUPS,
)


def get_loss_function(dataset):
    """
    Returns the appropriate loss function based on the dataset.

    :param dataset: A string that represents the name of the dataset
    :return: A string indicating the type of loss function 
    """
    if dataset in [NEWSGROUPS]:
        return "cos"
    if dataset in [MNIST, SCRNA]:
        return "L1"
    else:
        return "L2"


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
        # TODO: this is debugging settings
        num_data = 4000
        return np.linspace(1000, num_data, 4, dtype=int)

    else:
        num_data = 40000

    return np.linspace(10000, num_data, 4, dtype=int)


def run_scaling_experiment_with_k():
    """
    Runs scaling experiments varying the number of medoids (k) for the MNIST
    and CIFAR datasets using all BanditPAM algorithms.
    """
    # for dataset in [MNIST, CIFAR]: TODO uncomment this!
    for dataset in [NEWSGROUPS]:
        loss = get_loss_function(dataset)
        scaling_experiment_with_k(
            dataset_name=dataset,
            loss=loss,
            algorithms=ALL_BANDITPAMS,
            n_medoids_list=[5, 8, 10],
            dirname="20newsgroups"
        )


def run_scaling_experiment_with_n():
    """
    Runs scaling experiments varying the number of data points (n) for the
    provided datasets using all BanditPAM algorithms.
    """
    # TODO: run loop through all datasets
    for dataset in [NEWSGROUPS]:
        loss = get_loss_function(dataset)
        num_data_list = get_num_data_list(dataset)
        for n_medoids in [10]:
            np.random.seed(1)
            scaling_experiment_with_n(
                dataset_name=dataset,
                loss=loss,
                algorithms=ALL_BANDITPAMS,
                n_medoids=n_medoids,
                num_data_list=num_data_list,
                dirname="newsgroups_scaling_with_n",
                num_experiments=1,  # TODO: used to be 10
                parallelize=True,
            )


def run_debug_scrna():
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
    # run_scaling_experiment_with_k()
    run_scaling_experiment_with_n()