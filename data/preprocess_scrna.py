import pandas as pd
from sklearn.decomposition import TruncatedSVD

if __name__ == "__main__":
    dataset = pd.read_csv("scRNA_reformat.csv.gz", delimiter=",", header=None)

    # Create a TruncatedSVD object with n_components=4000 (desired dimension)
    svd = TruncatedSVD(n_components=4000)

    # Fit the TruncatedSVD model to your data and transform it to the reduced dimension
    reduced_data = svd.fit_transform(dataset)

    # 'reduced_data' now contains your dataset with dimensionality reduced to 4000
    # Convert the reduced_data array back to a DataFrame with column names if needed
    reduced_data_df = pd.DataFrame(reduced_data)

    # Save the reduced_data DataFrame to a CSV file
    reduced_data_df.to_csv("reduced_scrna.csv", header=False, index=False)

    print(reduced_data_df.shape)
    print(reduced_data_df.describe())
