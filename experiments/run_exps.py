import numpy as np
import time
import os

import banditpam
from create_configs import get_exp_name


def get_data(dataset: str, n: int, seed: int) -> np.ndarray:
    if dataset == "MNIST":
        data = np.loadtxt(os.path.join("..", "data", "MNIST_70k.csv"))
    elif dataset == "CIFAR":
        data = np.loadtxt(os.path.join("..", "data", "cifar10.csv"), delimiter=",")
    elif dataset == "SCRNA":
        data = np.loadtxt(os.path.join("..", "data", "reduced_scrna.csv"), delimiter=",")
    elif dataset == "NEWSGROUPS":
        data = np.loadtxt(os.path.join("..", "data", "20_newsgroups.csv"), delimiter=",", skiprows=1)  # Drop header
        data = data[:, 1:]  # Skip the first column, which is the datapoint index
    else:
        raise Exception("Bad dataset")

    np.random.seed(seed)
    return np.random.choice(data, size=n)

def run_exp(exp: dict) -> None:
    """
    Runs the given experiment.

    :param exp: The experiment to run.
    """
    # Check if the results for the experiment already exist
    exp_name = get_exp_name(exp)
    assert exp['parallelize'] == False, "Should only be running experiments with parallelize=False"
    if not os.path.exists(os.path.join("logs", exp_name)):
        # If they don't, run the experiment
        print(f"Running experiment {exp_name}...")
        if exp['algorithm'] == "BP++":
            algorithm = "BanditPAM"
            use_cache = True
        elif exp['algorithm'] == "BP+CA":
            algorithm = "BanditPAM_orig"
            use_cache = True
        elif exp['algorithm'] == "BP+VA":
            algorithm = "BanditPAM"
            use_cache = False
        elif exp['algorithm'] == "BP":
            algorithm = "BanditPAM_orig"
            use_cache = False

        kmed = banditpam.KMedoids(
            n_medoids=exp['k'],
            algorithm=algorithm,
            use_cache=use_cache,
            use_perm=use_cache,  # Use a permutation if and only if we use the cache
            max_iter=exp['T'],
            parallelize=exp['parallelize'],
            cache_width=exp['cache_width'],
            build_confidence=exp['build_confidence'],
            swap_confidence=exp['swap_confidence'],
            seed=exp['seed'],
        )

        # Fit on the dataset, loss, and seed
        data = get_data(exp['dataset'], exp['n'], exp['seed'])
        start = time.time()
        kmed.fit(data, exp['loss'])
        end = time.time()
        runtime = end - start


        # Query the key statistics and log to file
        # TODO: Get results
        # TODO: Implement querying of key statistics in BanditPAM and BanditPAM++
        # - BUILD sample complexity -- binding
        # - SWAP sample complexity -- binding
        # - Misc sample complexity -- binding
        # - Build Wall Clock Time - binding
        # - SWAP Wall Clock Time - binding
        # - Total Wall Clock Time - binding
        # - Number of swaps -- binding
        # - Build Loss -- binding
        # - Final Loss -- binding
        # - Build medoids -- binding
        # - Final medoids -- binding


    else:
        print(f"Already have results for {exp_name}...")

