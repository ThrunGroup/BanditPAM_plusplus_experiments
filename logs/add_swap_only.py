import os
import pandas as pd
from glob import glob

from scripts.constants import (
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_VA_CACHING,
    MNIST,
    CIFAR,
    SCRNA,
)

for algorithm in [BANDITPAM_ORIGINAL_NO_CACHING, BANDITPAM_VA_CACHING]:
    for dataset in [MNIST, CIFAR]:
        for n_medoids in [5, 10]:
            # for i in range(4):
            build_only_csv = pd.read_csv(
                os.path.join(
                    "build_only_all",
                    f"{algorithm}_{dataset}_k{n_medoids}_idx0.csv",
                )
            )
            build_only_time = build_only_csv["total_runtime"]

            csv_files = glob(
                os.path.join(
                    dataset,
                    f"{algorithm}_{dataset}_k{n_medoids}_idx*",
                )
            )

            for csv_file in csv_files:
                csv = pd.read_csv(csv_file)
                total_runtimes = csv["total_runtime"]
                average_runtimes = csv["runtime_per_swap"]
                num_swaps = csv["number_of_swaps"]
                swap_runtimes = total_runtimes - build_only_time

                print("Build only time: ", build_only_time)

                print(
                    "algorithm: ",
                    algorithm,
                    "\naverage: ",
                    average_runtimes,
                    "\nnew: ",
                    swap_runtimes,
                    "\ntotal_runtime: ",
                    total_runtimes,
                )
                break
            break
        break
    break
