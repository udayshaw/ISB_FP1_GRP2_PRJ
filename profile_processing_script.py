import pandas as pd
import numpy as np
import nltk
import regex as re
import spacy
from collections import Counter
from elasticsearch import Elasticsearch
import warnings
import os, json, traceback, logging
from elasticsearch.helpers import bulk
from processing_func import processing_engine
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

class data_processing:
    # Configure the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    pattern = r'[^a-zA-Z0-9\s\.\-\,]' #regex pattern to clean resumes and job description
    warnings.filterwarnings('ignore')
    stopwords = set(nltk.corpus.stopwords.words('english')) #fetching stopwords from ntlk
    nlp = spacy.load("en_core_web_sm") #fetching spacy library for english

    # Create a file handler 
    log_file = '/var/log/FP1_logs/log_file.log' #log file path
    file_handler = logging.FileHandler(log_file)

    # Configure the log file handler
    file_handler.setLevel(logging.INFO)

    # Create a formatter and configure it
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') #log format
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    def __init__(self):
        self.instance_var = "Instance Variable"

    #Function to clean the text and do NLP
    def do_nlp(self, text: str, profile_name : str=''):
        pe=processing_engine()
        self.logger.info('Processing text')
        content = list(self.nlp(re.sub(self.pattern, '', text)).sents)
        content_new = [[str(i)] for i in content]
        df=pd.DataFrame(content_new, columns=["resume_text"])
        ##############Computing tokens
        df["tokens"]=df[["resume_text"]].applymap(lambda x: pe.tokenize(x))
        ##############Computing bigrams
        df["bigrams"]=df[["resume_text"]].applymap(lambda x: pe.ngrams(pe.tokenize(x), 2, stopwords=self.stopwords))
        ##############Computing Trigrams
        df["trigrams"]=df[["resume_text"]].applymap(lambda x: pe.ngrams(pe.tokenize(x), 3, stopwords=self.stopwords))
        df["phrases"]=df["tokens"]+df["bigrams"]+df["trigrams"]
        df1=df[["phrases"]]
        ##############Computing term frequency(tf)
        freq_df=pe.count_words(df1,column="phrases")
        ##############Computing IDF
        idf_df = pe.compute_idf(df1,column="phrases")
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
        pe=processing_engine()
        self.logger.info('Initiating Processing profile')
        Base = declarative_base()

        class table_schema(Base):
            __tablename__ = pe.get_mySQLTable()
            profileId = Column(String, primary_key=True)
            resume = Column(String)

        Session = pe.get_mySQL_session()
        profile_name=fileName
        flag =0
        msg="Profile Processing status:<br>" #message to return
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
            msg=msg+"<br>MySQL load successfull."
        except Exception as e:
            flag=1
            msg=msg+"<br>MySQL load failed."
            self.logger.error("Adding data to MySQL failed with error: "+str(e))
            traceback.print_exc()
                
        ##############Processing data to put in Elastic     
        try:
            self.logger.debug('started Proceessing')
            freq_df=self.do_nlp(file_content,profile_name)

            ##############Pushing to elastic search
            self.logger.debug('Pushing Data to elastic Index')
            pe.push_to_elastic(freq_df)
            self.logger.debug('completed Proceessing')
            msg=msg+"<br>Elastic data load successfull"
            self.logger.info('File Proceessed')
        except Exception as e:
            flag=1
            msg=msg+"<br>Elastic data load failed"
            self.logger.error('Error processing resume with error: '+str(e))
            traceback.print_exc()

        if flag == 0:
            msg=msg+"<br><br>File processed successfully."
        else:
            msg=msg+"<br><br>File processing Failed."

        return msg

    #function to search a profile given a JD
    def get_profiles(self, jd :str):
        pe=processing_engine()
        self.logger.info('Looking up profiles')
        freq_df=self.do_nlp(jd)
        freq_df=freq_df[["token","tfidf"]]

        final_df=pd.concat([
            freq_df.sort_values("tfidf",ascending=False).head(5),
            freq_df[freq_df['token'].str.contains(' ')].sort_values("tfidf",ascending=False).head(5)]
            )
        key_list=final_df["token"].to_list()
        es = pe.connect_elastic()
        index_name = pe.get_elasticIndex()
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
        #Query Elastic search
        results = es.search(index=index_name, body=json.loads(query_body))

        df_list=[]

        for hit in results['hits']['hits']:
            df_list.append([hit['_source']['token'],hit['_source']['tfidf'],hit['_source']['profile_id']])
        #generating output table          
        df_out=pd.DataFrame(df_list,columns=["token","tfidf","profile_id"])
        #grouped = df_out.groupby('profile_id')['tfidf'].sum()
        grouped = df_out.groupby('profile_id').agg(tfidf=('tfidf','sum'), tokens=('token',lambda x: list(x)))
        sorted_descending = grouped.sort_values(by='tfidf', ascending=False).reset_index()
        sorted_descending["Download Link"]="/resumes_corpus/"+sorted_descending["profile_id"]+".txt"
        sorted_descending=sorted_descending.rename(columns={'tfidf': 'TF_IDF score'}). \
                                            rename(columns={'tokens': 'Keywords Matched'}). \
                                            rename(columns={'profile_id': 'Profile Id'})
        html_table=sorted_descending.head(10).to_html(index=False, escape=False, classes='table table-bordered table-striped')
        html_table = html_table.replace('<td>/resumes_corpus/', '<td><a href="/resumes_corpus/'). \
                        replace('.txt</td>', '.txt" target="_blank">download</td>'). \
                        replace('</td>', '</a></td>')
        return html_table