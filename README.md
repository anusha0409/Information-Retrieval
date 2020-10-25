Search Engine for Movies
--------------------------------------------------------------------------------------------------
***Domain specific Information Retrieval System***

**Problem Statement**:

The task is to build a search engine which will cater to the needs of a particular domain. You have
to feed your IR model with documents containing information about the chosen domain. It will
then process the data and build indexes. Once this is done, the user will give a query as an input.
You are supposed to return top 10 relevant documents as the output.

**About the project**

Dataset used - [Kaggle-movie-plots](https://www.kaggle.com/jrobischon/wikipedia-movie-plots)

Project By:
- **Kriti Jethlia**: Email- <f20180223@hyderabad.bits-pilani.ac.in>
- **Jui Pradhan**: Email- <f20180984@hyderabad.bits-pilani.ac.in>
- **Anusha Agarwal**: Email- <f20180032@hyderabad.bits-pilani.ac.in>
--------------------------------------------------------------------------------------------------
**How to run the code**
--------------------------------------------------------------------------------------------------

1. Clone the repository : https://github.com/anusha0409/Information-Retrieval.git
2. cd Information-Retrieval
3. Run files in the order: 

              python3 preprocess.py
              python3 tfidf.py
              python3 server.py
4. In your browser go to `http://0.0.0.0:3000/`
5. Type your query in the search bar and wait till it returns the relevant documents :)

---------------------------------------------------------------------------------------------------
**Dependencies/modules used**
---------------------------------------------------------------------------------------------------
- time
- nltk
- pandas
- pickle
- Numpy
- heapq
- flask
- os
