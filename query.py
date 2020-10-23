import time
import nltk
import pandas as pd
import numpy as np
import pickle
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from collections import defaultdict
import heapq

def load(doc):
    file = open(doc,'rb')
    df = pickle.load(file)
    file.close()
    return df

def query_processing(query):

    no_of_doc = 34886+1 # Actual number of documents + 1

    #all alphabets to lower case
    query= query.lower() 

    #preprocessing query removing unnecessary characters
    query = query.replace("\n"," ").replace("\r"," ")
    query = query.replace("'s"," ")
    punctuationList = '!"#$%&\()*+,-./:;<=>?@[\\]^_{|}~'
    x = str.maketrans(dict.fromkeys(punctuationList," "))
    query = query.translate(x)

    #tokenize
    df=word_tokenize(query)
    query_length=len(df)
    df= [w for w in df if not w in stopwords.words('english') ]
    
    #stemming
    ps = PorterStemmer() 
    df=[ps.stem(word) for word in df]
    
    #term frequency 
    query_freq = defaultdict(lambda: 0)

    for token in df :
        query_freq[token] +=1

    q_ls, q_tf_idf = helper("inverted_index.obj", "processed_data.obj", "Length", no_of_doc, query_length, query_freq)
    q_ls_title, q_tf_idf_title = helper("inverted_index_title.obj", "processed_data_title.obj", "TitleLength", no_of_doc, query_length, query_freq)

    #cosine similarity
    tf_idf = load("tf-idf.obj")
    tf_idf_title = load("tf-idf_title.obj")

    q_ls += q_ls_title
    q_ls = np.array(q_ls)
    norm_query= np.sqrt(np.sum(q_ls*q_ls))

    rank_heap=[]
    for doc in range(0,no_of_doc-1) :
        val=0   # dot-product
        epsilon=10e-9
        d_ls = [] #store tf-idf values for a doc
        for term , value in tf_idf[doc].items() :
            if (term in q_tf_idf.keys()):
                val+=(value * q_tf_idf[term])
            d_ls+=[value]
        for term , value in tf_idf_title[doc].items() :
            if (term in q_tf_idf_title.keys()):
                val+=(value * q_tf_idf_title[term])
            d_ls+=[value]
        d_ls = np.array(d_ls)
        norm_doc = np.sqrt(np.sum(d_ls*d_ls))
        cosine_sim = val/(norm_doc*norm_query + epsilon)
        heapq.heappush(rank_heap, (cosine_sim, doc))
    req_doc= heapq.nlargest(10, rank_heap)

    return req_doc




def helper(inverted_index, processed_data, length, no_of_doc, query_length, query_freq):
    #tf-idf for query

    ii_df = load(inverted_index)

    ii_df= ii_df.to_dict()
    ii_df=ii_df['PostingList']

    k=1.75 # parameter for BM25
    b=0.75 # parameter for BM25

    tdf = load(processed_data)
    

    avg_length= tdf[length].mean() #Average length of documents
    avg_length= ((no_of_doc-1)*avg_length + query_length)/no_of_doc

    q_tf_idf={} # Stores only the tf-idf values that are common with the bag of words
    q_ls=[] # Stores all the tf-idf values
    for key , value in query_freq.items() :
        tf = (value/query_length)
        if key in ii_df.keys(): 
            idf = np.log(no_of_doc/ii_df[key])
            # q_tf_idf[key]=tf*idf
            q_tf_idf[key] = idf*( tf*(k+1) )/(k*(1- b + b*(query_length/avg_length))) *100
        # q_ls=q_tf_idf[key]=tf*idf
        q_ls+= [np.log(no_of_doc)*( tf*(k+1) )/(k*(1- b + b*(query_length/avg_length))) *100] # add the value of tf idf of each term to the list

    return q_ls, q_tf_idf

# print(query_processing("bar appear enjoy face inside group work"))         
