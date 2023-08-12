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
    cache_width=50000,   # TODO: perhaps change to 10000 for speedup?
    parallelize=True,
    n_swaps=10,     # TODO: this used to be 100, but loss p much converges at 10..
    build_confidence=5,
    swap_confidence=100,
):
    if algorithm_name == BANDITPAM_ORIGINAL_NO_CACHING:
        algorithm = "BanditPAM_orig"
        use_cache = False
    elif algorithm_name == BANDITPAM_ORIGINAL_CACHING:
        algorithm = "BanditPAM_orig"
        use_cache = True
    elif algorithm_name == BANDITPAM_VA_CACHING:
        algorithm = "BanditPAM"
        use_cache = True
    elif algorithm_name == BANDITPAM_VA_NO_CACHING:
        algorithm = "BanditPAM"
        use_cache = False
    elif algorithm_name == FASTPAM:
        algorithm = "FastPAM1"
        use_cache = False
    else:
        assert False, "Incorrect algorithm!"

    kmed = banditpam.KMedoids(
        n_medoids=n_medoids,
        algorithm=algorithm,
        use_cache=use_cache,
        use_perm=use_cache,
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
