BANDITPAM_ORIGINAL_NO_CACHING = "BanditPAM Original without caching"
BANDITPAM_ORIGINAL_CACHING = "BanditPAM Original with caching"
BANDITPAM_VA_NO_CACHING = "BanditPAM VA without caching"
BANDITPAM_VA_CACHING = "BanditPAM VA with caching"
FASTPAM = "FastPAM1"

ALL_BANDITPAMS = [
    BANDITPAM_ORIGINAL_NO_CACHING,
    BANDITPAM_ORIGINAL_CACHING,
    BANDITPAM_VA_NO_CACHING,
    BANDITPAM_VA_CACHING,
]

# Graph
NUM_DATA = "num_data"
NUM_MEDOIDS = "num_medoids"
VAR_DELTA = "delta"
NUM_SWAPS = "num_swaps"

K_LIST = [5, 10, 15]

MNIST = "MNIST"
SCRNA = "SCRNA"
CIFAR = "CIFAR"
NEWSGROUPS = "NEWSGROUPS"

ALG_TO_COLOR = {
    BANDITPAM_ORIGINAL_NO_CACHING: "#8AB6D6",
    BANDITPAM_ORIGINAL_CACHING: "#0061A8",
    BANDITPAM_VA_NO_CACHING: "#FF9494",
    BANDITPAM_VA_CACHING: "#C21010",
}

ALG_TO_LABEL = {
    BANDITPAM_ORIGINAL_NO_CACHING: "BanditPAM",
    BANDITPAM_ORIGINAL_CACHING: "BanditPAM++ (No virtual arms)",
    BANDITPAM_VA_NO_CACHING: "BanditPAM++ (No caching)",
    BANDITPAM_VA_CACHING: "BanditPAM++",
}

SAMPLE_COMPLEXITY = "Sample Complexity"
RUNTIME = "total_runtime"
LOSS = "loss"

# varying delta corresponds to varying confidence intervals
SWAP_CONFIDENCE_ARR = [2, 3, 5, 10]

# 20 Newsgroups dataset size
NEWSGROUPS_NUM_DATA = 10000

LOSS_HISTORY = "loss_history"
