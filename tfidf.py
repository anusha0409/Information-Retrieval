import time
import nltk
import pandas as pd
import numpy as np
import pickle
''' Function to create tf-idr for all the documents '''
def tf_idf_preprocess():
    no_of_doc = 34886

    # loading term frequencies
    file = open("processed_data.obj",'rb')
    df = pickle.load(file)
    file.close()
    df.rename( columns={'Unnamed: 0':'Document'}, inplace=True)
    d_df= df.to_dict()
    d_df=d_df['Length']
    

    # average length of all documents
    avg_length= df["Length"].mean() 
    
    #loading indexing list
    file = open("inverted_index.obj",'rb')
    ii_df = pickle.load(file)
    file.close()

    ii_df.rename( columns={'Unnamed: 0':'Words'}, inplace=True )
    ii_df.set_index('Words',inplace=True)
    ii_df= ii_df.to_dict()
    ii_df=ii_df['PostingList'] 

    k=1.75 # parameter for BM25
    b=0.75 # parameter for BM25
    
    tf_idf_dict={}
    start_time = time.time()

    #calculating tf-idf
    for doc in range(0,no_of_doc):
        doc_dict={}
        for key,value in eval(df['Frequency'][doc]).items():
            if key=='nan' or key=='null':
             continue;
            tf = (value/d_df[doc]) 
            idf = np.log(no_of_doc/(ii_df[key]))
            # doc_dict[key]=tf*idf
            doc_dict[key] = idf*( tf*(k+1) )/(k*(1- b + b*(df["Length"][doc]/avg_length))) * 100
        tf_idf_dict[doc]=doc_dict

    print("--- %s seconds ---" % (time.time() - start_time))
    return tf_idf_dict  

tf_idf_dict = tf_idf_preprocess()

filehandler = open("tf-idf.obj","wb")
pickle.dump(tf_idf_dict,filehandler)
filehandler.close()




