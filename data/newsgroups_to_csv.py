import sklearn
import pandas as pd
from sentence_transformers import SentenceTransformer
from scripts.constants import NEWSGROUPS_NUM_DATA

""" 
Preprocesses the 20newsgroups dataset into a .csv file to use in banditPAM
NOTE: get sentence_transformers with <pip install -U >
"""

def twenty_newsgroup_to_csv(num_data=NEWSGROUPS_NUM_DATA):
    newsgroups_train = sklearn.datasets.fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
    )
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Embedding 20 Newsgroups data...")
    embedding = model.encode(newsgroups_train.data[:num_data])
    df = pd.DataFrame(embedding.reshape(-1, embedding.shape[1]).tolist())
    print("Saving 20 Newsgroups to csv...")
    # the first row is just the indices of the embeddings
    df.to_csv('20_newsgroups.csv')


if __name__ == "__main__":
    twenty_newsgroup_to_csv()