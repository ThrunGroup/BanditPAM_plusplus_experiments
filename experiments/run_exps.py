import numpy as np
import os

import banditpam
from create_configs import get_exp_name, get_exp_params_from_name

MACHINE_INDEX = 0
NUM_MACHINES = 19

def get_data(dataset: str, n: int, seed: int) -> np.ndarray:
    if dataset == "MNIST":
        data = np.loadtxt(os.path.join("..", "data", "MNIST_70k.csv"))
    elif dataset == "CIFAR10":
        data = np.loadtxt(os.path.join("..", "data", "cifar10.csv"), delimiter=",")
    elif dataset == "SCRNA":
        data = np.loadtxt(os.path.join("..", "data", "reduced_scrna.csv"), delimiter=",")
    elif dataset == "NEWSGROUPS":
        data = np.loadtxt(os.path.join("..", "data", "20_newsgroups.csv"), delimiter=",", skiprows=1)  # Drop header
        data = data[:, 1:]  # Skip the first column, which is the datapoint index
    else:
        raise Exception("Bad dataset")

    np.random.seed(seed)
    return data[np.random.choice(len(data), size=n)]

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

        )
        kmed.seed = exp['seed']  # Not supported as a constructor argument. This calls setSeed().

        # Fit on the dataset, loss, and seed
        data = get_data(exp['dataset'], exp['n'], exp['seed'])
        kmed.fit(data, exp['loss'])


        # Query the key statistics and log to file
        with open(os.path.join("logs", exp_name), "w+") as fout:
            fout.write("Build distance comps: " + str(kmed.build_distance_computations) + "\n")
            fout.write("Swap distance comps: " + str(kmed.swap_distance_computations) + "\n")
            fout.write("Misc distance comps: " + str(kmed.misc_distance_computations) + "\n")
            fout.write("Build + Swap distance comps: " + str(kmed.getDistanceComputations(False)) + "\n")
            fout.write("Total distance comps: " + str(kmed.getDistanceComputations(True)) + "\n")
            fout.write("Number of Steps: " + str(kmed.steps) + "\n")
            fout.write("Total Build time: " + str(kmed.total_build_time) + "\n")
            fout.write("Total Swap time: " + str(kmed.total_swap_time) + "\n")
            fout.write("Time per swap: " + str(kmed.time_per_swap) + "\n")
            fout.write("Total time: " + str(kmed.total_time) + "\n")
            fout.write("Build loss: " + str(kmed.build_loss) + "\n")
            fout.write("Final loss: " + str(kmed.average_loss) + "\n")
            fout.write("Loss trajectory: " + str(kmed.losses) + "\n")
            fout.write("Build medoids: " + str(kmed.build_medoids) + "\n")
            fout.write("Final medoids: " + str(kmed.medoids) + "\n")
            fout.write("Cache Hits: " + str(kmed.cache_hits) + "\n")
            fout.write("Cache Misses: " + str(kmed.cache_misses) + "\n")
            fout.write("Cache Writes: " + str(kmed.cache_writes) + "\n")

    else:
        print(f"Already have results for {exp_name}...")

def main():
    with open("all_configs.csv", "r") as exp_file:
        for line_idx, line in enumerate(exp_file):
            if line_idx % NUM_MACHINES == MACHINE_INDEX:
                exp = get_exp_params_from_name(line.strip())
                run_exp(exp)


if __name__ == "__main__":
    main()