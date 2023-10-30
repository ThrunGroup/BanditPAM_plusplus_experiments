DATASETS_AND_LOSSES = [("MNIST", "L2"), ('CIFAR10', "L1"), ('SCRNA', "L1"), ('NEWSGROUPS', "cos")]
DATASETS_AND_LOSSES_WITHOUT_SCRNA = [("MNIST", "L2"), ('CIFAR10', "L1"), ('NEWSGROUPS', "cos")]
ALL_ALGORITHMS = ["BP++", "BP+CA", "BP+VA", "BP"]
ALGORITHM_TO_LEGEND = {
    "BP++": "BP++",
    "BP+CA": "BP+PIC",
    "BP+VA": "BP+VA",
    "BP": "BP",
}

ALG_COLORS = {
    "BP++": "crimson",
    "BP+VA": "gray",
    "BP+CA": "pink",
    "BP": "black",
}

ALG_LINESTYLES = {
    "BP++": "solid",
    "BP+VA": "solid",
    "BP+CA": "solid",
    "BP": "solid",
}