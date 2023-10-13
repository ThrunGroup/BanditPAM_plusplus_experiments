import os
import pandas as pd

files = os.listdir()

for file in files:
    print(file)
    if file in [
        "BanditPAM VA with caching_MNIST_k5_idx0.csv",
        "BanditPAM VA with caching_SCRNA_n10000_idx0.csv",
        "remove_loss_history.py",
    ]:
        continue
    try:
        csv = pd.read_csv(file)
        csv = csv.drop("loss_history", axis=1)
        csv["average_swap_runtime"] = csv["average_swap_runtime: "]
        csv = csv.drop("average_swap_runtime: ", axis=1)
        csv.to_csv(file, index=False)
    except:
        continue
