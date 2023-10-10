from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sentence_transformers import SentenceTransformer
from scripts.constants import NEWSGROUPS_NUM_DATA

""" 
Preprocesses the 20newsgroups dataset into a .csv file to use in banditPAM
NOTE: get sentence_transformers with <pip install -U sentence-transformers>
"""

def twenty_newsgroup_to_csv(num_data=NEWSGROUPS_NUM_DATA):
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
    )
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("=> embedding data")
    embedding = model.encode(newsgroups_train.data[:num_data])
    df = pd.DataFrame(embedding.reshape(-1, embedding.shape[1]).tolist())
    print("=> saving to csv")
    df.to_csv('data/20_newsgroups.csv')   # the first row is just the indices of the embeddings

    