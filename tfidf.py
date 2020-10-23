import time
import nltk
import pandas as pd
import numpy as np

def tf_idf_preprocess():
    no_of_doc = 10
    df = pd.read_csv('Processed_Data.csv')
    avg_length= df["Length"].mean()
    ii_df = pd.read_csv('Inverted_Index.csv')
    ii_df.rename( columns={'Unnamed: 0':'Words'}, inplace=True )
    df.rename( columns={'Unnamed: 0':'Document'}, inplace=True )
    k=1.75 # parameter for B25
    b=0.75 # parameter for B25
    tf_idf_df = pd.DataFrame(0, index = range(len(ii_df)),columns=range(no_of_doc))
    tf_idf_df  = pd.concat([ii_df[['Words']],tf_idf_df], axis=1)
    tf_idf_df.set_index('Words', inplace=True)
    ii_df.set_index('Words',inplace=True)
    for doc in range(0,9):
        for key,value in eval(df['Frequency'][doc]).items():
            tf = (value/df["Length"][doc]) 
            idf = np.log(no_of_doc/ii_df.loc[key, "PostingList"])
            # tf_idf_df.loc[key,doc] = tf*idf*100  # normal tf-idf
            tf_idf_df.loc[key,doc] =( tf*(k+1) )/(k*(1- b + b*(df["Length"][doc]/avg_length))) * 100
    return tf_idf_df   
tf_idf_df = tf_idf_preprocess()
tf_idf_df.to_csv(r'Doc_tf_idf.csv')