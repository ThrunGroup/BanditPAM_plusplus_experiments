import pickle
import os
import pandas as pd


def load_cifar_batch(file):
    """
    Function to load a single batch file of the CIFAR-10 dataset.
    """
    with open(file, "rb") as f:
        dict = pickle.load(f, encoding="bytes")

    return dict


if __name__ == "__main__":
    # Define the directory where the data batches are stored.
    # Assumes we are being run from the /data directory
    DIRECTORY = "cifar-10-batches-py"

    # List of data batch files to load; there are 5 shards
    batches = ["data_batch_" + str(i) for i in range(1, 6)]

    dataframes = []
    for batch in batches:
        batch_data = load_cifar_batch(os.path.join(DIRECTORY, batch))
        df = pd.DataFrame(batch_data[b"data"])
        df["label"] = batch_data[b"labels"]
        dataframes.append(df)

    # Combine all batch dataframes
    df = pd.concat(dataframes)
    df /= 255.0  # Normalize pixel values

    # Write to CSV
    df.to_csv("cifar10.csv", header=False, index=False)

    print(df.shape)
    print(df.describe())
