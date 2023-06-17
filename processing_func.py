import pandas as pd
import numpy as np
import nltk
import regex as re
import spacy
from collections import Counter
from elasticsearch import Elasticsearch
import warnings
import os, json
from elasticsearch.helpers import bulk
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

class processing_engine:
    def __init__(self):
        self.instance_var = "Instance Variable"

    #function to tokenize
    def tokenize(self, text):
        return re.findall(r'[\w-]*\p{L}[\w-]*', text.lower())

    #function to compute IDF
    def compute_idf(self, df, column='tokens', preprocess=None, min_df=2):
        def update(doc):
            tokens = doc if preprocess is None else preprocess(doc) 
            counter.update(set(tokens))
        # count tokens
        counter = Counter()
        df[column].map(update)
        
        # create DataFrame and compute idf
        idf_df = pd.DataFrame.from_dict(counter, orient='index', columns=['df'])
        idf_df = idf_df.query('df >= @min_df')
        idf_df['idf'] = np.log(len(df)/idf_df['df'])+0.1
        idf_df.index.name = 'token'
        return idf_df

    #function to compute ngrams    
    def ngrams(self, tokens, n=2, sep=' ', stopwords=set()):
        return [sep.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)]) if len([t for t in ngram if t in stopwords])==0]

    #function to compute TF
    def count_words(self, df, column='tokens', preprocess=None, min_freq=2):
        # process tokens and update counter
        def update(doc):
            tokens = doc if preprocess is None else preprocess(doc)
            counter.update(tokens)
        # create counter and run through all data
        counter = Counter()
        df[column].map(update)
        # transform counter into a DataFrame
        freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
        freq_df = freq_df.query('freq >= @min_freq')
        freq_df.index.name = 'token'
        return freq_df.sort_values('freq', ascending=False)
    
    #get elastic connection
    def connect_elastic(self):
        es=Elasticsearch("http://localhost:9200")
        return es

    #get elastic Index
    def get_elasticIndex(self):
        return 'processed_profiles'
    
    #get MySQL Table
    def get_mySQLTable(self):
        return 'input_resumes'
    
    #get MySQL session
    def get_mySQL_session(self):
        # Create a SQLAlchemy engine and session
        engine = create_engine('mysql+pymysql://group2:isbfp1@localhost/isb_term2_fp1')
        Session = sessionmaker(bind=engine)
        return Session

    #write to elastic index
    def push_to_elastic(self, freq_df):
        es = self.connect_elastic()
        index_name = self.get_elasticIndex()
        data = freq_df.to_dict(orient='records')
        actions = [
            {
                '_index': index_name,
                '_source': document
            }
            for document in data
        ]
        # Use the bulk API to index the documents
        bulk(es, actions)
        # Refresh the index
        es.indices.refresh(index=index_name)
