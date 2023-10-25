import numpy as np
from typing import List

from constants import (
    DATASETS_AND_LOSSES,
    ALL_ALGORITHMS,
)

def str_to_bool(s: str) -> bool:
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError(f"String {s} cannot be converted to bool")


def get_exp_name(exp: dict) -> str:
    return str(
        exp['dataset'] + '_' +
        exp['loss'] + '_' +
        exp['algorithm'] + '_' +
        str(exp['n']) + '_' +
        str(exp['k']) + '_' +
        str(exp['T']) + '_' +
        str(exp['cache_width']) + '_' +
        str(exp['build_confidence']) + '_' +
        str(exp['swap_confidence']) + '_' +
        str(exp['parallelize']) + '_' +
        str(exp['seed'])
    )


def get_exp_params_from_name(exp_name: str) -> dict:
    tokens = exp_name.split('_')
    return {
        'dataset': tokens[0],
        'loss': tokens[1],
        'algorithm': tokens[2],
        'n': int(tokens[3]),
        'k': int(tokens[4]),
        'T': int(tokens[5]),
        'cache_width': int(tokens[6]),
        'build_confidence': int(tokens[7]),
        'swap_confidence': int(tokens[8]),
        'parallelize': str_to_bool(tokens[9]),
        'seed': int(tokens[10]),
    }


def get_table_1_exps() -> List:
    exps = []
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
                    exps.append(exp_name)
    return exps


def get_figures_1_and_2_exps() -> List:
    exps = []
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
                    exps.append(exp_name)
    return exps


def get_figure_3_exps() -> List:
    exps = []
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
                    exps.append(exp_name)
    return exps


def get_appendix_table_1_exps() -> List:
    exps = []
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
                    exps.append(exp_name)
    return exps


def get_appendix_figure_1_exps() -> List:
    exps = []
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
                    exps.append(exp_name)
    return exps


def get_appendix_table_2_exps() -> List:
    exps = []
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
                    exps.append(exp_name)
    return exps


def add_exps(exps: List, new_exps: List):
    for exp in new_exps:
        if exp not in exps:
            exps.append(exp)


def generate_config():
    """
    Generates the config file for all experiments from the paper.
    """
    added_exps = []

    # For Table 1
    table_1_exps = get_table_1_exps()
    add_exps(added_exps, table_1_exps)

    # For Figures 1 and 2
    figures_1_and_2_exps = get_figures_1_and_2_exps()
    add_exps(added_exps, figures_1_and_2_exps)

    # For Figure 3
    figure_3_exps = get_figure_3_exps()
    add_exps(added_exps, figure_3_exps)

    # For Appendix Table 1
    appendix_table_1_exps = get_appendix_table_1_exps()
    add_exps(added_exps, appendix_table_1_exps)

    # For Appendix Figure 1
    appendix_figure_1_exps = get_appendix_figure_1_exps()
    add_exps(added_exps, appendix_figure_1_exps)

    # For Appendix Table 2
    appendix_table_2_exps = get_appendix_table_2_exps()
    add_exps(added_exps, appendix_table_2_exps)

    # Write all configs to file
    with open("all_configs.csv", "w+") as fout:
        for exp_name in added_exps:
            fout.write(exp_name + '\n')

if __name__ == "__main__":
    generate_config()



