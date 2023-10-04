import banditpam
import time

from scripts.constants import (
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_ORIGINAL_CACHING,
    BANDITPAM_VA_NO_CACHING,
    BANDITPAM_VA_CACHING,
    FASTPAM,
)


def run_banditpam(
    algorithm_name,
    data,
    n_medoids,
    loss,
    cache_width=2000,
    parallelize=True,
    n_swaps=100,
    build_confidence=10,
    swap_confidence=6,
):
    """
    Runs the given algorithm.

    :param algorithm_name: The name of the algorithm to run.
    :param data: The data to run the algorithm on.
    :param n_medoids: The number of medoids to use.
    :param loss: The loss function to use.
    :param cache_width: The width of the cache to use.
    :param parallelize: Whether or not to parallelize the algorithm.
    :param n_swaps: The number of swaps to use.
    :param build_confidence: The confidence to use during the BUILD step.
    :param swap_confidence: The confidence to use during the SWAP step.
    :return: The algorithm object and the runtime.

    """
    if algorithm_name == BANDITPAM_ORIGINAL_NO_CACHING:
        algorithm = "BanditPAM_orig"
        use_cache = False
    elif algorithm_name == BANDITPAM_ORIGINAL_CACHING:
        algorithm = "BanditPAM_orig"
        use_cache = True
    elif algorithm_name == BANDITPAM_VA_NO_CACHING:
        algorithm = "BanditPAM"
        use_cache = False
    elif algorithm_name == BANDITPAM_VA_CACHING:
        algorithm = "BanditPAM"
        use_cache = True
    elif algorithm_name == FASTPAM:
        algorithm = "FastPAM1"
        use_cache = False
    else:
        raise Exception("Incorrect algorithm!")

    kmed = banditpam.KMedoids(
        n_medoids=n_medoids,
        algorithm=algorithm,
        use_cache=use_cache,
        use_perm=use_cache,  # Use a permutation if and only if we use the cache
        max_iter=n_swaps,
        parallelize=parallelize,
        cache_width=cache_width,
        build_confidence=build_confidence,
        swap_confidence=swap_confidence,
    )
    start = time.time()
    kmed.fit(data, loss)
    end = time.time()
    runtime = end - start

    return kmed, runtime
