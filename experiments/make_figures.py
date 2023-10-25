import numpy as np
import pandas as pd

from constants import (
    DATASETS_AND_LOSSES,
    ALL_ALGORITHMS,
)

def generate_config():
    """
    Generates the config file for all experiments from the paper.
    """
    added_exps = []





def make_table_1():
    for n in [10000, 15000, 20000, 25000, 30000]:
        for dataset, loss in DATASETS_AND_LOSSES:
            if dataset in ["MNIST", "CIFAR10"]:
                k = 10
            elif dataset in ["SCRNA", "NEWSGROUPS"]:
                k = 5

            T = 10
            cache_width = 1000
            build_confidence = 3
            swap_confidence = 5
            parallelize = False

            for algorithm in ["BP++", "BP"]:
                for seed in range(1):
                    exp = {
                        'dataset': dataset,
                        'loss': loss,
                        'algorithm': algorithm,
                        'n': n,
                        'k': k,
                        'T': T,
                        'cache_width': cache_width,
                        'build_confidence': build_confidence,
                        'swap_confidence': swap_confidence,
                        'parallelize': parallelize,
                        'seed': seed,
                    }
                    exp_name = get_exp_name(exp)
                    if exp_name not in added_exps:
                        added_exps.append(exp_name)

def make_figures_1_and_2():
    # For Fig 1 and Fig 2

    for dataset, loss in DATASETS_AND_LOSSES:
        if dataset in ["MNIST", "CIFAR10"]:
            k = 10
        elif dataset in ["SCRNA", "NEWSGROUPS"]:
            k = 5

        if dataset == "MNIST":
            n_schedule = np.linspace(10000, 70000, 5, dtype=int)
        elif dataset == "CIFAR10":
            n_schedule = np.linspace(10000, 30000, 5, dtype=int)
        elif dataset == "SCRNA":
            n_schedule = np.linspace(10000, 40000, 4, dtype=int)
        elif dataset == "NEWSGROUPS":
            n_schedule = np.linspace(10000, 50000, 5, dtype=int)
        else:
            raise Exception("Bad dataset")

        T = 10
        cache_width = 1000
        build_confidence = 3
        swap_confidence = 5
        parallelize = False

        for n in n_schedule:
            for algorithm in ALL_ALGORITHMS:
                for seed in range(3):
                    exp = {
                        'dataset': dataset,
                        'loss': loss,
                        'algorithm': algorithm,
                        'n': n,
                        'k': k,
                        'T': T,
                        'cache_width': cache_width,
                        'build_confidence': build_confidence,
                        'swap_confidence': swap_confidence,
                        'parallelize': parallelize,
                        'seed': seed,
                    }
                    exp_name = get_exp_name(exp)
                    if exp_name not in added_exps:
                        added_exps.append(exp_name)




def make_figure_3():
    for dataset, loss in DATASETS_AND_LOSSES:
        k_schedule = [5, 10, 15]

        if dataset in ["MNIST", 'CIFAR10']:
            n = 20000
        elif dataset in ["SCRNA", "NEWSGROUPS"]:
            n = 10000
        else:
            raise Exception("Bad dataset")

        T = 10
        cache_width = 1000
        build_confidence = 3
        swap_confidence = 5
        parallelize = False

        for k in k_schedule:
            for algorithm in ALL_ALGORITHMS:
                for seed in range(3):
                    exp = {
                        'dataset': dataset,
                        'loss': loss,
                        'algorithm': algorithm,
                        'n': n,
                        'k': k,
                        'T': T,
                        'cache_width': cache_width,
                        'build_confidence': build_confidence,
                        'swap_confidence': swap_confidence,
                        'parallelize': parallelize,
                        'seed': seed,
                    }
                    exp_name = get_exp_name(exp)
                    if exp_name not in added_exps:
                        added_exps.append(exp_name)

def make_appendix_table_1():
    # For Appendix Table 1

    for dataset, loss in DATASETS_AND_LOSSES:
        n = 10000

        if dataset in ["MNIST", 'CIFAR10']:
            k = 10
        elif dataset in ["SCRNA", "NEWSGROUPS"]:
            k = 5
        else:
            raise Exception("Bad dataset")

        T = 10
        cache_width = 1000
        build_confidence = 3
        parallelize = False

        swap_confidence_schedule = [2, 3, 5, 10]

        for swap_confidence in swap_confidence_schedule:
            for algorithm in ["BP++", "BP"]:
                for seed in range(3):
                    exp = {
                        'dataset': dataset,
                        'loss': loss,
                        'algorithm': algorithm,
                        'n': n,
                        'k': k,
                        'T': T,
                        'cache_width': cache_width,
                        'build_confidence': build_confidence,
                        'swap_confidence': swap_confidence,
                        'parallelize': parallelize,
                        'seed': seed,
                    }
                    exp_name = get_exp_name(exp)
                    if exp_name not in added_exps:
                        added_exps.append(exp_name)


def make_appendix_figure_1():
    # For Appendix Figure 1

    for dataset, loss in DATASETS_AND_LOSSES:
        n = 10000

        if dataset in ["MNIST", 'CIFAR10']:
            k = 10
        elif dataset in ["SCRNA", "NEWSGROUPS"]:
            k = 5
        else:
            raise Exception("Bad dataset")

        T_schedule = range(1, 11)
        cache_width = 1000
        build_confidence = 3
        swap_confidence = 5
        parallelize = False

        for T in T_schedule:
            for algorithm in ["BP++", "BP"]:
                for seed in range(3):
                    exp = {
                        'dataset': dataset,
                        'loss': loss,
                        'algorithm': algorithm,
                        'n': n,
                        'k': k,
                        'T': T,
                        'cache_width': cache_width,
                        'build_confidence': build_confidence,
                        'swap_confidence': swap_confidence,
                        'parallelize': parallelize,
                        'seed': seed,
                    }
                    exp_name = get_exp_name(exp)
                    if exp_name not in added_exps:
                        added_exps.append(exp_name)



def make_appendix_table_2():
    # For Appendix Table 2

    for dataset, loss in DATASETS_AND_LOSSES:
        n = 10000
        k_schedule = [5, 10, 15]
        T = 10
        cache_width = 1000
        build_confidence = 3
        swap_confidence = 5
        parallelize = False

        for k in k_schedule:
            for algorithm in ["BP++", "BP"]:
                for seed in range(3):
                    exp = {
                        'dataset': dataset,
                        'loss': loss,
                        'algorithm': algorithm,
                        'n': n,
                        'k': k,
                        'T': T,
                        'cache_width': cache_width,
                        'build_confidence': build_confidence,
                        'swap_confidence': swap_confidence,
                        'parallelize': parallelize,
                        'seed': seed,
                    }
                    exp_name = get_exp_name(exp)
                    if exp_name not in added_exps:
                        added_exps.append(exp_name)

def make_all_figures():
    make_table_1()
    make_figures_1_and_2()
    make_figure_3()
    make_appendix_table_1()
    make_appendix_figure_1()
    make_appendix_table_2()


if __name__ == "__main__":
    make_all_figures()