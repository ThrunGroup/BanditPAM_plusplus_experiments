import numpy as np
import pandas as pd
import os
from typing import List
# import matplotlib
# matplotlib.use("GTK3Agg")

import matplotlib.pyplot as plt


from constants import (
    DATASETS_AND_LOSSES,
    DATASETS_AND_LOSSES_WITHOUT_SCRNA,
    ALL_ALGORITHMS,
    ALGORITHM_TO_LEGEND,
    ALG_COLORS,
    ALG_LINESTYLES,
)

from create_configs import (
    str_to_bool,
    get_exp_name,
    get_exp_params_from_name,
    get_table_1_exps,
    get_figures_1_and_2_exps,
    get_figure_3_exps,
    get_appendix_table_1_exps,
    get_appendix_figure_1_exps,
    get_appendix_table_2_exps,
)

FIG_SIZE = (16, 16)

def parse_logfile(logfile: str) -> dict:
    with open(logfile, 'r') as f:
        lines = f.readlines()
        result = {}
        for line in lines:
            try:
                result[line.split(':')[0].strip()] = line.split(':')[1].strip()
            except IndexError as _e:
                ## This can happen, for example, because the arrays are split onto multiple lines, like:
                #Build medoids: [17476  3500  4876 16042  6786 19663 15019  3366  8387  7888  3316 12897
                #  5529  4493  19409]
                # We don't need to use the loss trajectories, build medoids, or final medoids, so skip
                assert line[:1] == ' ', "Line should start with a space, but is_{}".format(line)
                result[line.split(':')[0].strip()] = ''
        return result

HEADERS = [
    # These MUST be the same as the keys in the exp_parms dicts
    "dataset",
    "loss",
    "algorithm",
    "n",
    "k",
    "T",
    "cache_width",
    "build_confidence",
    "swap_confidence",
    "parallelize",
    "seed",

    # These MUST be the same as the rows in the logfiles
    'Build distance comps',
    'Swap distance comps',
    'Misc distance comps',
    'Build + Swap distance comps',
    'Total distance comps',
    'Number of Steps',
    'Total Build time',
    'Total Swap time',
    'Time per swap',
    'Total time',
    'Build loss',
    'Final loss',
    'Loss trajectory',
    'Build medoids',
    'Final medoids',
    'Cache Hits',
    'Cache Misses',
    'Cache Writes',
]

def get_pd_from_exps(exps: List[str]) -> pd.DataFrame:
    results = pd.DataFrame(columns=HEADERS)
    for exp_idx, exp in enumerate(exps):
        exp_params = get_exp_params_from_name(exp)
        logfile = os.path.join("logs", exp)
        if os.path.exists(logfile):
            exp_result = parse_logfile(logfile)

            # The | takes the second dict's values.
            # If there are any conflicts. In this case, there shouldn't be any, so it's a merge of dicts
            row_dict = exp_params | exp_result

            # I'm so sorry
            row = pd.DataFrame.from_dict([row_dict])
            results = pd.concat([results, row])

    return results

def make_table_1():
    table_1_exps = get_table_1_exps()
    table_1_results = get_pd_from_exps(table_1_exps)

    # For 4 datasets and 4 sizes of n and each seed, measure loss row of BP++ over BP
    table_1_results = table_1_results[['dataset', 'n', 'seed', 'algorithm', 'Final loss']]
    table_1_results_bp = table_1_results[table_1_results['algorithm'] == 'BP']
    table_1_results_bppp = table_1_results[table_1_results['algorithm'] == 'BP++']
    merged = table_1_results_bppp.merge(table_1_results_bp, on=['dataset', 'n', 'seed'], how='left')
    answer1 = pd.to_numeric(merged['Final loss_x'])
    answer2 = pd.to_numeric(merged['Final loss_y'])

    answer = answer1 / answer2
    merged['ratio'] = answer  # Should get 1.0 everywhere
    print(merged)

def make_figures_1_and_2():
    fig1_2_exps = get_figures_1_and_2_exps()
    fig1_2_results = get_pd_from_exps(fig1_2_exps)

    # Set title
    dataset_to_title = {
        "MNIST": "MNIST",
        "CIFAR10": "CIFAR10",
        "SCRNA": "scRNA",
        "NEWSGROUPS": "20 Newsgroups",
    }

    loss_to_title = {
        "L1": "$L_1$",
        "L2": "$L_2$",
        "cos": "Cosine",
    }

    ks = {
        "MNIST": 10,
        "CIFAR10": 10,
        "SCRNA": 5,
        "NEWSGROUPS": 5,
    }

    def make_individual_figure(fig, ax, dataset, loss, xlabel, ylabel, filename):
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax.set_title(f"{dataset_to_title[dataset]}, {loss_to_title[loss]}, $k = {ks[dataset]}$")
        ax.grid()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
        fig.savefig(os.path.join("figures", filename), format='pdf')

    for dataset_idx in range(len(DATASETS_AND_LOSSES_WITHOUT_SCRNA)):
        fig1, ax1 = plt.subplots(1, 1)
        fig2, ax2 = plt.subplots(1, 1)
        dataset, loss = DATASETS_AND_LOSSES_WITHOUT_SCRNA[dataset_idx]
        for algorithm_idx, algorithm in enumerate(ALL_ALGORITHMS):
            algo_results = fig1_2_results[(fig1_2_results['algorithm'] == algorithm) & (fig1_2_results['dataset'] == dataset) & (fig1_2_results['loss'] == loss)]
            xs = pd.to_numeric(algo_results['n'])
            times = pd.to_numeric(algo_results['Total time']) / (1000*(pd.to_numeric(algo_results['Number of Steps']) + 1))
            samples = pd.to_numeric(algo_results['Total distance comps']) / (pd.to_numeric(algo_results['Number of Steps']) + 1)

            ax1.plot(xs, times, label=ALGORITHM_TO_LEGEND[algorithm], marker='o', linestyle=ALG_LINESTYLES[algorithm], color=ALG_COLORS[algorithm])
            ax2.plot(xs, samples, label=ALGORITHM_TO_LEGEND[algorithm], marker='o', linestyle=ALG_LINESTYLES[algorithm], color=ALG_COLORS[algorithm])

        make_individual_figure(fig1, ax1, dataset, loss, "Subsample size ($n$)", "Time per step (s)", f"figure1_{dataset}.pdf")
        make_individual_figure(fig2, ax2, dataset, loss, "Subsample size ($n$)", "Sample complexity per step", f"figure2_{dataset}.pdf")



def make_figure_3():
    fig3_exps = get_figure_3_exps()
    fig3_results = get_pd_from_exps(fig3_exps)

    # Set title
    dataset_to_title = {
        "MNIST": "MNIST",
        "CIFAR10": "CIFAR10",
        "SCRNA": "scRNA",
        "NEWSGROUPS": "20 Newsgroups",
    }

    loss_to_title = {
        "L1": "$L_1$",
        "L2": "$L_2$",
        "cos": "Cosine",
    }

    ns = {
        "MNIST": 20000,
        "CIFAR10": 20000,
        "SCRNA": 10000,
        "NEWSGROUPS": 10000,
    }

    def make_individual_figure(fig, ax, dataset, loss, xlabel, ylabel, filename):
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_title(f"{dataset_to_title[dataset]}, {loss_to_title[loss]}, $n = {ns[dataset]}$")
        ax.grid()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
        fig.savefig(os.path.join("figures", filename), format='pdf')

    for dataset_idx in range(len(DATASETS_AND_LOSSES_WITHOUT_SCRNA)):
        fig3, ax3 = plt.subplots(1, 1)
        fig4, ax4 = plt.subplots(1, 1)
        dataset, loss = DATASETS_AND_LOSSES_WITHOUT_SCRNA[dataset_idx]
        for algorithm_idx, algorithm in enumerate(ALL_ALGORITHMS):
            algo_results = fig3_results[
                (fig3_results['algorithm'] == algorithm) & (fig3_results['dataset'] == dataset) & (
                            fig3_results['loss'] == loss)]
            xs = pd.to_numeric(algo_results['k'])
            times = pd.to_numeric(algo_results['Total time']) / (1000 * (pd.to_numeric(algo_results['Number of Steps']) + 1))
            samples = pd.to_numeric(algo_results['Total distance comps']) / (pd.to_numeric(algo_results['Number of Steps']) + 1)

            ax3.plot(xs, times, label=ALGORITHM_TO_LEGEND[algorithm], marker='o', linestyle=ALG_LINESTYLES[algorithm],
                     color=ALG_COLORS[algorithm])
            ax4.plot(xs, samples, label=ALGORITHM_TO_LEGEND[algorithm], marker='o', linestyle=ALG_LINESTYLES[algorithm],
                     color=ALG_COLORS[algorithm])

        make_individual_figure(fig3, ax3, dataset, loss, "Number of medoids ($k$)", "Time per step (s)", f"figure3_{dataset}.pdf")
        make_individual_figure(fig4, ax4, dataset, loss, "Number of medoids ($k$)", "Sample complexity per step", f"figure4_{dataset}.pdf")



def make_appendix_table_1():
    appendix_table_1_exps = get_appendix_table_1_exps()
    a_table_1_results = get_pd_from_exps(appendix_table_1_exps)

    # For 4 datasets and 4 sizes of n and each seed, measure loss row of BP++ over BP
    a_table_1_results = a_table_1_results[['dataset', 'n', 'seed', 'algorithm', 'swap_confidence', 'Final loss']]
    a_table_1_results_bp = a_table_1_results[a_table_1_results['algorithm'] == 'BP']
    a_table_1_results_bppp = a_table_1_results[a_table_1_results['algorithm'] == 'BP++']
    merged = a_table_1_results_bppp.merge(a_table_1_results_bp, on=['dataset', 'n', 'seed', 'swap_confidence'], how='left')
    answer1 = pd.to_numeric(merged['Final loss_x'])
    answer2 = pd.to_numeric(merged['Final loss_y'])

    answer = answer1 / answer2
    merged['ratio'] = answer  # Should get 1.0 everywhere
    print(merged)


def make_appendix_figure_1():
    afig1_exps = get_appendix_figure_1_exps()
    afig1_results = get_pd_from_exps(afig1_exps)



    # Set title
    dataset_to_title = {
        "MNIST": "MNIST",
        "CIFAR10": "CIFAR10",
        "SCRNA": "scRNA",
        "NEWSGROUPS": "20 Newsgroups",
    }

    loss_to_title = {
        "L1": "$L_1$",
        "L2": "$L_2$",
        "cos": "Cosine",
    }

    ks = {
        "MNIST": 10,
        "CIFAR10": 10,
        "SCRNA": 5,
        "NEWSGROUPS": 5,
    }

    for dataset_idx in range(len(DATASETS_AND_LOSSES_WITHOUT_SCRNA)):
        afig1, axa1 = plt.subplots(1, 1)
        dataset, loss = DATASETS_AND_LOSSES_WITHOUT_SCRNA[dataset_idx]
        algorithm = "BP++"
        algo_results = afig1_results[
            (afig1_results['algorithm'] == algorithm) & (afig1_results['dataset'] == dataset) & (
                    afig1_results['loss'] == loss)]
        xs = (pd.to_numeric(algo_results['T'])).astype(int)

        losses = pd.to_numeric(algo_results['Final loss'])
        axa1.plot(xs, losses, label=algorithm, marker='o')
        axa1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axa1.ticklabel_format(axis='x', style='plain')
        axa1.set_title(
            f"{dataset_to_title[dataset]}, {loss_to_title[loss]}, $n = 10000$, $k = {ks[dataset]}$")
        axa1.grid()
        axa1.set_ylabel("Final clustering loss")
        axa1.set_xlabel("Number of SWAP iterations ($T$)")
        afig1.savefig(os.path.join("figures", "appendix_figure1_" + str(dataset) + ".pdf"), format='pdf')


def make_appendix_table_2():
    print("Making appendix table 2...")
    appendix_table_2_exps = get_appendix_table_2_exps()
    a_table_2_results = get_pd_from_exps(appendix_table_2_exps)

    # For 4 datasets and 4 sizes of n and each seed, measure loss row of BP++ over BP
    a_table_2_results = a_table_2_results[['dataset', 'n', 'k', 'seed', 'algorithm', 'Total Build time', 'Total time', 'Time per swap']]
    a_table_2_results_bp = a_table_2_results[a_table_2_results['algorithm'] == 'BP']
    a_table_2_results_bppp = a_table_2_results[a_table_2_results['algorithm'] == 'BP++']
    merged = a_table_2_results_bppp.merge(a_table_2_results_bp, on=['dataset', 'n', 'k', 'seed'], how='left')

    bppp_build_time = pd.to_numeric(merged['Total Build time_x'])
    bp_build_time = pd.to_numeric(merged['Total Build time_y'])
    build_ratio = bp_build_time / bppp_build_time
    merged['build speedup ratio'] = build_ratio

    bppp_swap_time = pd.to_numeric(merged['Time per swap_x'])
    bp_swap_time = pd.to_numeric(merged['Time per swap_y'])
    swap_ratio = bp_swap_time / bppp_swap_time
    merged['swap speedup ratio'] = swap_ratio

    bppp_total_time = pd.to_numeric(merged['Total time_x'])
    bp_total_time = pd.to_numeric(merged['Total time_y'])
    total_ratio = bp_total_time / bppp_total_time
    merged['total speedup ratio'] = total_ratio


    print(merged)

def make_all_figures():
    make_table_1()
    make_figures_1_and_2()
    make_figure_3()
    make_appendix_table_1()
    make_appendix_figure_1()
    make_appendix_table_2()


if __name__ == "__main__":
    make_all_figures()