import os
import pandas as pd
from glob import glob
import numpy as np

from scripts.constants import (
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_VA_CACHING,
    MNIST,
    CIFAR,
    SCRNA,
)

files = os.listdir()

for dataset in [SCRNA]:
    for n_medoids in [5, 10, 15]:
        average_total_runtimes = []
        average_swap_runtimes = []

        for algorithm in [BANDITPAM_ORIGINAL_NO_CACHING, BANDITPAM_VA_CACHING]:
            path = glob(
                f"{algorithm}_{dataset}_k{n_medoids}_idx*",
            )[0]
            csv = pd.read_csv(path)
            average_total_runtime = csv["average_runtime"]
            # average_swap_runtime = csv["average_swap_runtime"]
            average_total_runtimes += (average_total_runtime,)
            # average_swap_runtimes += (average_swap_runtime,)

        speedup = average_total_runtimes[0] / average_total_runtimes[1]
        # swap_speedup = average_swap_runtimes[0] / average_swap_runtimes[1]

        average_speedup = speedup
        # average_swap_speedup = np.mean(swap_speedup)
        print(dataset, n_medoids)
        print(
            average_speedup)
        # average_swap_speedup)
        # average_swap_speedup / average_speedup,
