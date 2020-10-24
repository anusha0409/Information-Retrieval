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
        The goal is to create an indexing list and store it in a pickle file.
        """
        self.tokenizer_w = WhitespaceTokenizer()
        self.stop = stopwords.words('english')
        self.ps = PorterStemmer()
        
    def read(self):
        '''
        Reads the dataset which is in csv format and stores it as a dataframe in df.
        Returns the dataframe df.
        '''
        df = pd.read_csv('wiki_movie_plots_deduped.csv')
        filehandler = open("movie_plot.obj","wb")
        pickle.dump(df,filehandler)
        filehandler.close()
        return df
    
    def LowerCase(self, df):
        '''
        Converts the text to lower case and replaces NA with ''.
        Takes parameters as the dataframe and returns the processed dataframe 'df'
        '''
        
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
        '''
        Removes punctuations and escape sequences.
        Takes the text to be modified as a parameter and returns the modified text.
        '''
        
        text = text.replace("\n"," ").replace("\r"," ")
        text = text.replace("'s"," ")
        punctuationList = '!"#$%&\()*+,-./:;<=>?@[\\]^_{|}~'
        x = str.maketrans(dict.fromkeys(punctuationList," "))
        text = text.translate(x)
        return text
        
    def tokenizeHelper(self, text):
        '''
        Calls the nltk WhiteSpaceTokenizer to tokenize.
        Takes parameter as the text and returns the tokenized text.
        '''
        
        text = self.preprocess(text)
        return self.tokenizer_w.tokenize(text)

    def Tokenizer(self, df):
        '''
        Adds Columns to the dataframe containing the tokens.
        Takes parameter as the dataframe and returns the dataframe with columns containing the tokens.
        '''
        
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
        '''
        This Function removes the stopwords from the Tokens Column in the DataFrame.
        Takes the dataframe as a parameter. The changed dataframe is returned.
        '''
        
        print("Time required to Remove Stop Words")
        self.start_time = time.time()
        df['Tokens'] = df['Tokens'].apply(lambda x: [item for item in x if item not in self.stop])
        print("--- %s seconds ---" % (time.time() - self.start_time))
        return df

    def Stemmer(self, df, x):
        '''
        This Function uses Porter's Stemmer for Stemming.
        Takes the dataframe and the column name as parameters.
        Stemming is done on the column name x.
        Function returns the dataframe 'df'.
        '''
        
        print("Time required for Stemming")
        self.start_time = time.time()
        df['stemmed'] = df[x].apply(lambda x: [self.ps.stem(word) for word in x])
        print("--- %s seconds ---" % (time.time() - self.start_time))
        return df

    def BagOfWords(self, uniqueWords, tokens):
        '''
        Creates a Dictionary with Keys as words and Values as the word-frequency in the document.
        Function parameter : the dataframe column 'Unique_Words' (which stores the unique words per document).
                           : the dataframe column 'Tokens' (which stores the tokens per document).
        Function returns a dictionary called numOfWords.
        '''
        
        unique = tuple(uniqueWords)
        numOfWords = dict.fromkeys(unique, 0)
        for word in tokens:
            numOfWords[word] += 1
        return numOfWords

    def TermFrequency(self, df_tokenized):
        '''
        Calculates the term frequency of each word document-wise.
        Function takes the dataframe as parameters and returns a dataframe with extra columns.
        'Unique _Words' column stores unique words per document by using set.
        'Frequency' column stores the frequency of each word per document(i.e term frequency).
        '''
        
        print("Time required to create the Term Frequency")
        self.start_time = time.time()
        
        df_tokenized['Unique_Words'] = df_tokenized['stemmed'].apply(set)
        df_tokenized['Frequency'] = df_tokenized.apply(lambda x: self.BagOfWords(x.Unique_Words, x.stemmed), axis=1)
        print("--- %s seconds ---" % (time.time() - self.start_time))
        return df_tokenized
    
    def Vocabulary(self, df_tokenized):
        '''
        Creates Vocabulary for all the documents. i.e Stores all the unique tokens.
        Takes dataframe as a parameter and returns the unique words in the entire dataset(stored as Inverted_Index).
        Uses a set to calculate unique words for the entire dataset.
        '''
        
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
        '''
        Adds The posting list to the Inverted Index DataFrame.
        Takes the indexing list and the dataframe as parameters and returns the indexing list.
        '''
        
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

    def Write(self, file, df):
        '''
        Stores the dataframes as pickle files.
        Takes filename and dataframe as parameter.
        '''
        filehandler = open(file,"wb")
        pickle.dump(df,filehandler)
        filehandler.close()
    
    def main(self):
        df = self.read()
        df = self.LowerCase(df)
        df = self.Tokenizer(df)
        df = self.RemoveStopWords(df)
        df = self.Stemmer(df, 'Tokens')
        df = self.TermFrequency(df)
        # Stores only the required columns from the processed dataframe.
        df1 = df[['Length' , 'Frequency']]
        # Storing inverted processed data as pickle
        self.Write("processed_data.obj", df1)
        
        df_Title = df[['TokensTitle', 'TitleLength']]
        df_Title = self.Stemmer(df_Title, 'TokensTitle')
        df_Title = self.TermFrequency(df_Title)
        # Storing inverted processed data as pickel
        self.Write("processed_data_title.obj", df_Title)


        Inverted_Index = self.Vocabulary(df)
        Inverted_Index = self.InvertedIndex(Inverted_Index, df)
        # Storing inverted index as pickle
        self.Write("inverted_index.obj", Inverted_Index)

        Inverted_Index_Title = self.Vocabulary(df_Title)
        Inverted_Index_Title = self.InvertedIndex(Inverted_Index_Title, df_Title)
        # Storing inverted index of Title as pickle
        self.Write("inverted_index_title.obj", Inverted_Index_Title)       

Data = ProcessData()
Data.main()
