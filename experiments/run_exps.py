import numpy as np
import time
import os

import banditpam
from create_configs import get_exp_name


# TODO: Implement get_data
def get_data(dataset: str, n: int, seed: int) -> np.ndarray:
    if dataset == "MNIST":
        pass
    elif dataset == "CIFAR":
        pass
    elif dataset == "SCRNA":
        pass
    elif dataset == "NEWSGROUPS":
        pass
    else:
        raise Exception("Bad dataset")

    np.random.seed(seed)
    return np.random.choice(dataset, size=n)
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

        # TODO: Set random seed
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
        # TODO: Log to file
        # TODO: Implement querying of key statistics in BanditPAM and BanditPAM++

    else:
        print(f"Already have results for {exp_name}...")

