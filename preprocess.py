# Importing Libraries

import time
import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle


class ProcessData:
    
    def __init__(self):
        """
        This Class is used to process the Dataset, followed by Normalization and Tokenization.
        The goal is to create an inverted index and store it in csv format
        """
        self.tokenizer_w = WhitespaceTokenizer()
        self.stop = stopwords.words('english')
        self.ps = PorterStemmer()
        
    def read(self):
        df = pd.read_csv('wiki_movie_plots_deduped.csv')
        filehandler = open("movie_plot.obj","wb")
        pickle.dump(df,filehandler)
        filehandler.close()
        return df
    
    def LowerCase(self, df):
        
        print("Time required for Preprocessing and Tokenizing")
        self.start_time = time.time()
        # Remove NA values
        df = df.fillna('')
        
        # 'data' variable stores column names used to form the corpus
        data = ['Plot','Title','Origin/Ethnicity', 'Director', 'Cast', 'Genre' ]
        
        # Convert all text to Lower Case
        for item in data:
            df[item] = df[item].str.lower()
        df = df.fillna('')
        return df
        
    def preprocess(self, text):
        '''Removes punctuations and escape sequences'''
        
        text = text.replace("\n"," ").replace("\r"," ")
        text = text.replace("'s"," ")
        punctuationList = '!"#$%&\()*+,-./:;<=>?@[\\]^_{|}~'
        x = str.maketrans(dict.fromkeys(punctuationList," "))
        text = text.translate(x)
        return text
        
    def tokenizeHelper(self, text):
        '''Calls the nltk WhiteSpaceTokenizer to tokenize'''
        
        text = self.preprocess(text)
        return self.tokenizer_w.tokenize(text)

    def Tokenizer(self, df):
        '''Adds Columns to the dataframe containing the tokens'''
        
        df['TokensPlot'] = df['Plot'].apply(self.tokenizeHelper)
        df['TokensTitle'] = df['Title'].apply(self.tokenizeHelper)
        df['TokensOrigin'] = df['Origin/Ethnicity'].apply(self.tokenizeHelper)
        df['TokensDirector'] = df['Director'].apply(self.tokenizeHelper)
        df['TokensCast'] = df['Cast'].apply(self.tokenizeHelper)
        df['TokensGenre'] = df['Genre'].apply(self.tokenizeHelper)
        
        # Tokens column stores the tokens for the corresponding document
        df['Tokens'] = df['TokensPlot'] + df['TokensTitle'] + df['TokensOrigin'] + df['TokensDirector'] + df['TokensCast'] + df['TokensGenre']
        df['Length'] = df.Tokens.apply(len)
        df['TitleLength'] = df.TokensTitle.apply(len)
        print("--- %s seconds ---" % (time.time() - self.start_time))
        return df
    
    def RemoveStopWords(self, df):
        '''This Function removes the stopwords from the Tokens Column in the DataFrame'''
        
        print("Time required to Remove Stop Words")
        self.start_time = time.time()
        df['Tokens'] = df['Tokens'].apply(lambda x: [item for item in x if item not in self.stop])
        print("--- %s seconds ---" % (time.time() - self.start_time))
        return df

    def Stemmer(self, df, x):
        '''This Function uses Porter's Stemmer for Stemming'''
        
        print("Time required for Stemming")
        self.start_time = time.time()
        df['stemmed'] = df[x].apply(lambda x: [self.ps.stem(word) for word in x])
        print("--- %s seconds ---" % (time.time() - self.start_time))
        return df

    def BagOfWords(self, uniqueWords, tokens):
        '''Creates a Dictionary with Keys as words and Values as the word-frequency in the document'''
        
        unique = tuple(uniqueWords)
        numOfWords = dict.fromkeys(unique, 0)
        for word in tokens:
            numOfWords[word] += 1
        return numOfWords

    def TermFrequency(self, df_tokenized):
        '''Calculates the term frequency of each word document-wise'''
        
        print("Time required to create the Term Frequency")
        self.start_time = time.time()
        
        df_tokenized['Unique_Words'] = df_tokenized['stemmed'].apply(set)
        df_tokenized['Frequency'] = df_tokenized.apply(lambda x: self.BagOfWords(x.Unique_Words, x.stemmed), axis=1)
        print("--- %s seconds ---" % (time.time() - self.start_time))
        return df_tokenized
    
    def Vocabulary(self, df_tokenized):
        '''Creates Vocabulary for all the documents. i.e Stores all the unique tokens'''
        
        print("Time required to create the Inverted Index")
        self.start_time = time.time()
        
        Inverted_Index = pd.DataFrame()
        tokens = set(df_tokenized['Unique_Words'][0])
        for i in range (0, 34885):
            tokens = set.union(tokens,set(df_tokenized['Unique_Words'][i+1]))
        Inverted_Index = pd.DataFrame(tokens)
        Inverted_Index.columns =['Words']
        return Inverted_Index
    
    def InvertedIndex(self, Inverted_Index, df_tokenized):
        '''Adds The posting list to the Inverted Index DataFrame'''
        
        inverted_index_dict = {}
        for i in range (0, 34886):
            for item in df_tokenized['Unique_Words'][i]:
                if item in inverted_index_dict.keys():
                    inverted_index_dict[item]+=1
                else:
                    inverted_index_dict[item]=1
           
        Inverted_Index = pd.Series(inverted_index_dict).to_frame()

        Inverted_Index.columns =['PostingList']
        print("--- %s seconds ---" % (time.time() - self.start_time))
        return Inverted_Index
    
    def main(self):
        df = self.read()
        df = self.LowerCase(df)
        df = self.Tokenizer(df)
        df = self.RemoveStopWords(df)
        df = self.Stemmer(df, 'Tokens')
        df = self.TermFrequency(df)
        # print(df)
        df1 = df[['Length' , 'Frequency']]

        
        df_Title = df[['TokensTitle', 'TitleLength']]
        df_Title = self.Stemmer(df_Title, 'TokensTitle')
        df_Title = self.TermFrequency(df_Title)

        # storing inveted processed data as pickel
        filehandler = open("processed_data.obj","wb")
        pickle.dump(df1,filehandler)
        filehandler.close()

        # storing inveted processed data as pickel
        filehandler = open("processed_data_title.obj","wb")
        pickle.dump(df_Title,filehandler)
        filehandler.close()

        # df.to_csv(r'Processed_Data.csv')

        Inverted_Index = self.Vocabulary(df)
        Inverted_Index = self.InvertedIndex(Inverted_Index, df)

        Inverted_Index_Title = self.Vocabulary(df_Title)
        Inverted_Index_Title = self.InvertedIndex(Inverted_Index_Title, df_Title)
        
        #storing inverted index as pickel
        filehandler = open("inverted_index.obj","wb")
        pickle.dump(Inverted_Index,filehandler)
        filehandler.close()

        #storing inverted index of Title as pickel
        filehandler = open("inverted_index_title.obj","wb")
        pickle.dump(Inverted_Index_Title,filehandler)
        filehandler.close()

        file = open("processed_data.obj",'rb')
        ii_df = pickle.load(file)
        file.close()
        # print(ii_df)

        file = open("processed_data_title.obj",'rb')
        ii_df1 = pickle.load(file)
        file.close()
        # print(ii_df1)
        # Inverted_Index.to_csv(r'Inverted_Index.csv')

        # df= pd.read_csv('wiki_movie_plots_deduped.csv')
        # filehandler = open("movie_data.obj","wb")
        # pickle.dump(df,filehandler)
        # filehandler.close()
        

Data = ProcessData()
Data.main()
