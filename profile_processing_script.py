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
from processing_func import processing_engine
import logging
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

class data_processing:
    # Configure the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    pattern = r'[^a-zA-Z0-9\s\.\-\,]'
    warnings.filterwarnings('ignore')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    nlp = spacy.load("en_core_web_sm")

    # Create a file handler
    log_file = 'log_file.log'
    file_handler = logging.FileHandler(log_file)

    # Configure the log file handler
    file_handler.setLevel(logging.INFO)

    # Create a formatter and configure it
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    def __init__(self):
        self.instance_var = "Instance Variable"

    def do_nlp(self, text: str, profile_name : str=''):
        self.logger.info('Processing text')
        content = list(self.nlp(re.sub(self.pattern, '', text)).sents)
        content_new = [[str(i)] for i in content]
        df=pd.DataFrame(content_new, columns=["resume_text"])
        ##############Computing tokens
        df["tokens"]=df[["resume_text"]].applymap(lambda x: processing_engine.tokenize(x))
        ##############Computing bigrams
        df["bigrams"]=df[["resume_text"]].applymap(lambda x: processing_engine.ngrams(processing_engine.tokenize(x), 2, stopwords=self.stopwords))
        ##############Computing Trigrams
        df["trigrams"]=df[["resume_text"]].applymap(lambda x: processing_engine.ngrams(processing_engine.tokenize(x), 3, stopwords=self.stopwords))
        df["phrases"]=df["tokens"]+df["bigrams"]+df["trigrams"]
        df1=df[["phrases"]]
        ##############Computing term frequency(tf)
        freq_df=processing_engine.count_words(df1,column="phrases")
        ##############Computing IDF
        idf_df = processing_engine.compute_idf(df1,column="phrases")
        ##############Computing TF-IDF
        freq_df['tfidf'] = freq_df['freq'] * idf_df['idf']
        ##############Adding profile Id to DF
        freq_df['profile_id']=profile_name
        freq_df=freq_df[['tfidf','profile_id']]
        freq_df.reset_index(inplace=True)
        ##############Removing stopword entires
        freq_df=freq_df[~freq_df['token'].isin(self.stopwords)].dropna()
        return freq_df

    def process_content(self, fileName:str, file_content:str):
        Base = declarative_base()

        class table_schema(Base):
            __tablename__ = 'input_resumes_test'

            profileId = Column(String, primary_key=True)
            resume = Column(String)

        Session = processing_engine.get_mySQL_session()

        cnt=0
        
        profile_name=fileName
        msg=""
        ##############Processing data to put in MySQL   
        try:
            self.logger.debug('Pushing to Mysql Started')
            session = Session()
            data_r = table_schema(profileId=profile_name,resume=re.sub(self.pattern, '', file_content))
            ##############Pushing to Mysql
            session.add(data_r)
            session.commit()
            session.close()
            self.logger.debug('Pushing to Mysql completed for')
            msg="MySQL load successfull."
        except Exception as e:
            msg="MySQL load failed."
            self.logger.error("Adding data to MySQL failed with error\n"+str(e))
                
        ##############Processing data to put in Elastic     
        try:
            self.logger.debug('started Proceessing')
            freq_df=do_nlp(file_content,profile_name)

            ##############Pushing to elastic search
            self.logger.debug('Pushing Data to elastic Index')
            processing_engine.push_to_elastic(freq_df)
            self.logger.debug('completed Proceessing')
            msg=msg+"<br>Elastic data load successfull"
            cnt=cnt+1
            self.logger.info('Files Proceessed: '+str(cnt))
        except Exception as e:
            msg=msg+"<br>Elastic data load failed"
            self.logger.error('Error processing resume with error\n'+str(e))

        return msg

    def get_profiles(self, jd :str):
        freq_df=self.do_nlp(jd)
        freq_df=freq_df[["token","tfidf"]]

        final_df=pd.concat([
            freq_df.sort_values("tfidf",ascending=False).head(5),
            freq_df[freq_df['token'].str.contains(' ')].sort_values("tfidf",ascending=False).head(5)]
            )
        key_list=final_df["token"].to_list()
        es = processing_engine.connect_elastic()
        index_name = processing_engine.get_elasticIndex()
        # Construct the Elasticsearch query body
        query_body = {
        "size": 1000,
        "query": {
            "terms": {
            "token.keyword": key_list
            }
        }
        }
        query_body=str(query_body).replace("'","\"")
        results = es.search(index=index_name, body=json.loads(query_body))

        df_list=[]

        for hit in results['hits']['hits']:
            df_list.append([hit['_source']['token'],hit['_source']['tfidf'],hit['_source']['profile_id']])
            
        df_out=pd.DataFrame(df_list,columns=["token","tfidf","profile_id"])
        grouped = df_out.groupby('profile_id')['tfidf'].sum()
        sorted_descending = grouped.sort_values(ascending=False).reset_index()
        return sorted_descending.head(10).to_html(index=False)