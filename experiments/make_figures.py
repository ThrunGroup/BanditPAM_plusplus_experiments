import numpy as np
import pandas as pd
import os
from typing import List

from constants import (
    DATASETS_AND_LOSSES,
    ALL_ALGORITHMS,
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
            result[line.split(':')[0].strip()] = line.split(':')[1].strip()
        print(result)
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
    "build_onfidence",
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
    print(table_1_results)



def make_figures_1_and_2():
    pass


def make_figure_3():
    pass

def make_appendix_table_1():
    pass


def make_appendix_figure_1():
    pass


def make_appendix_table_2():
    pass

def make_all_figures():
    make_table_1()
    make_figures_1_and_2()
    make_figure_3()
    make_appendix_table_1()
    make_appendix_figure_1()
    make_appendix_table_2()


if __name__ == "__main__":
    make_all_figures()