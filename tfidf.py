import time
import nltk
import pandas as pd
import numpy as np
import pickle
''' Function to create tf-idr for all the documents '''
def load(doc):
    file = open(doc,'rb')
    df = pickle.load(file)
    file.close()
    return df

def tf_idf_preprocess(processed_data, inverted_index, length):
    no_of_doc = 34886

    # loading term frequencies
    df = load(processed_data)

    d_df = df.to_dict()
    d_df =d_df[length]
    

    # average length of all documents
    avg_length= df[length].mean() 
    
    #loading indexing list
    ii_df = load(inverted_index)

    ii_df= ii_df.to_dict()
    ii_df=ii_df['PostingList'] 

    k=1.75 # parameter for BM25
    b=0.75 # parameter for BM25
    
    tf_idf_dict={}
    start_time = time.time()

    #calculating tf-idf
    for doc in range(0,no_of_doc):
        doc_dict={}
        for key,value in df['Frequency'][doc].items():
            if key=='nan' or key=='null':
             continue;
            tf = (value/d_df[doc]) 
            idf = np.log(no_of_doc/(ii_df[key]))
            # doc_dict[key]=tf*idf
            doc_dict[key] = idf*( tf*(k+1) )/(k*(1- b + b*(df[length][doc]/avg_length))) * 100
        tf_idf_dict[doc]=doc_dict

    print("--- %s seconds ---" % (time.time() - start_time))
    return tf_idf_dict  

tf_idf_dict = tf_idf_preprocess("processed_data.obj", "inverted_index.obj", "Length")
filehandler = open("tf-idf.obj","wb")
pickle.dump(tf_idf_dict,filehandler)
filehandler.close()

tf_idf_title_dict = tf_idf_preprocess("processed_data_title.obj", "inverted_index_title.obj", "TitleLength")
filehandler = open("tf-idf_title.obj","wb")
pickle.dump(tf_idf_title_dict,filehandler)
filehandler.close()

# print(tf_idf_dict)
# print(tf_idf_title_dict)





