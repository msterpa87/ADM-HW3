from nltk.stem.porter import PorterStemmer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from urllib.request import urlretrieve
from bs4 import BeautifulSoup as bs
from langdetect import detect
from collections import Counter
from math import log
from bs4 import BeautifulSoup
from urllib.request import urlopen
import datetime
import heapq
import pandas as pd
import numpy as np
import pickle
import re
from os import listdir, mkdir, stat
from os.path import exists, isdir
import sys
import string
import calendar


############### CONSTANTS ###############

BESTBOOK_URL = "https://www.goodreads.com/list/show/1.Best_Books_Ever?page="
BASE_URL = "https://www.goodreads.com"
HEADER_LIST = ["bookTitle","bookSeries","bookAuthors","ratingValue",
               "ratingCount","reviewCount","Plot","NumberofPages",
               "PublishingDate","Characters","Setting","Url"]
URL_FILENAME = "books"
TSV_FILENAME = "dataset.tsv"
vocabulary_filename = "vocabulary"
INDEX_FILENAME = "index"
TFIDF_FILENAME = "tfidf_index"
DEFAULT_COLS = ['bookTitle','Plot','Url'] # column visualized in queries
MONTHS = list(map(str.lower, calendar.month_name[1:]))
VERBOSE = False # set to True for console outputs
N_RESULTS = 10 # number of results to visualize in query
MAX_PAGE = 300
MIN_SIZE = 100*1024 # minimum html file size
CHUNK_SIZE = 1


def last_article(dirname=None):
    """
    Returns the number of the last article available in the specified directory
    or in the last directory available if not specified
    """

    if dirname is None:
        try:
            dirname = sorted(list(filter(lambda x: "page" in x, listdir())))[-1]
        except Exception as e:
            if VERBOSE: "[last_article()]:" + str(e)
            return 0

    files = list(filter(lambda x: "article" in x, listdir(dirname)))

    try:
        article_n = sorted(list(map(lambda x:int(x.split("_")[1].split(".")[0]),files)))[-1]
    except Exception as e:
        if VERBOSE: "[last_article()]:" + str(e)
        return 0

    return article_n    

def bookslist_from_file():
    """
    Returns a list of all the scraped urls
    """
    books_url = []

    with open(URL_FILENAME,"r") as file:
        books_url = file.read().splitlines()

    return books_url

def dirname_from_article(article_n):
    # return the pathname of an article given its number
    return "page_{}".format(article_n // 100 + 1)

def article_to_str(article_n):
    # return article string given its number
    return "article_{}.{}".format(article_n, 'html')

def article_to_page_num(article_n):
    # return the page number given an article number
    page_n = article_n // 100 + 1

    if article_n % 100 == 0:
        page_n -= 1

    return page_n

def article_to_page_str(article_n):
    return "page_{}".format(article_to_page_num(article_n))

def checkdir(dirname):
    # create directory if it doesn't exist
    if not isdir(dirname):
        mkdir(dirname)



############### CRAWLER FUNCTIONS ###############


def download_article(book_list, article_n):
    # Download the article html code into the appropriate directory
    article = article_to_str(article_n)
    page = article_to_page_str(article_n)
    pathname = "{}/{}".format(page, article)
    url = book_list[article_n - 1]
    urlretrieve(url, pathname)

def download_missing():
    # download missing html files
    book_list = bookslist_from_file()

    for i in range(1, 301):
        dirname = "page_" + str(i)

        if i % 10 == 0 and VERBOSE: print("page {}".format(i))

        for j in range((i - 1) * 100 + 1, i * 100):
            filename = "article_{}.html".format(j)
            pathname = dirname + "/" + filename

            # download file if not present or too small (likely corrupted)
            if not exists(pathname) or stat(pathname).st_size < MIN_SIZE:
                if VERBOSE: print("[{}] Download...".format(j))
                download_article(book_list, j)

def download_missing(dirname):
    # download missing html files
    book_list = bookslist_from_file()
    page_n = int(dirname.split("_")[1])

    for i in range((page_n - 1) * 100 + 1, page_n * 100):
        filename = "article_{}.html".format(i)
        pathname = dirname + "/" + filename

        # download file if not present or too small (likely corrupted)
        if not exists(pathname) or stat(pathname).st_size < MIN_SIZE:
            if VERBOSE: print("[{}] Download...".format(i))
            download_article(book_list, i)

def to_date(string):
    """
    takes the content of a PublishingDate html tag and clean it
    returning a valida datetime object
    """

    try:
        string = re.sub(r'\(|\)|first|published|','',string.lower())
    except Exception as e:
        if VERBOSE: print(e)
        return None
    
    regex = r'([a-z]+) ([0-9]{,2})[a-z]{2}? ([0-9]{4})|([a-z]+) ([0-9]{4})|([0-9]{4})'
    m = re.search(regex, string)
    try:
        pieces = list(filter(lambda x: x is not None, m.groups()))
    except Exception as e:
        if VERBOSE: print(e)
        return None
    
    n = len(pieces)
    year = int(pieces[-1])
    
    # when month or day is not in the string assume 1
    if n >= 2:
        try:
            month = MONTHS.index(pieces[0]) + 1
        except:
            return None
        if n == 2:
            day = 1
        else:
            day = int(pieces[1])
    if n == 1:
        month = 1
        day = 1
        
    return datetime.date(year=year, month=month, day=day)

def get_attribute(soup, attr):
    """
    Given a BeautifulSoup object and an attribute name returns its cleaned content
    """
    if attr == "plot":
        return soup.find('div', {'id':'description'}).text.strip()
    elif attr == "title":
        return soup.find('title').text.strip()
    elif attr == "bookseries":
        return soup.find('h2', {'id':'bookSeries'}).text.strip().strip("()")
    elif attr == "author":
        return soup.find('a', {'class':'authorName'}).text.strip()
    elif attr == "rating_value":
        return soup.find('span', {'itemprop':'ratingValue'}).text.strip()
    elif attr == "rating_count":
        return soup.find('meta', {'itemprop':'ratingCount'}).text.split()[0]
    elif attr == "review_count":
        return soup.find('meta', {'itemprop':'reviewCount'}).text.split()[0]
    elif attr == "pages":
        return soup.find('span', {'itemprop':'numberOfPages'}).text.split()[0]
    elif attr == "publishing_date":
        tags = soup.find('div',{'id':'details'}).find_all('div',{'class':'row'})[-1]
        return " ".join(tags.text.split())
    elif attr == "characters":
        tags = soup.find_all('a', {'href':re.compile("/characters")})
        return ", ".join(list(map(lambda x: x.text, tags)))
    elif attr == "places":
        tags = soup.find_all('a', {'href':re.compile("places")})
        return ", ".join(list(map(lambda x:x.text, tags)))

def soup_from_filename(pathname):
    # Returns a soup object associated with the content of the specified file
    with open(pathname,'r', encoding="utf8") as f:
        html = f.read()
        soup = bs(html,'html.parser')

    return soup


def crawler(book_list, start, end):
    """
    Takes a list of urls, start and end number of articles to download
    and saves them into the appropriate directory
    """
    dirname = dirname_from_article(start)

    checkdir(dirname)

    download_missing(dirname)

    last = last_article(dirname)

    if last > start:
        start = last

    for i in range(start, len(book_list[:end])):

        # change directory each 100 articles
        if i % 100 == 0 and i:
            dirname = dirname_from_article(i)
            checkdir(dirname)

        url = book_list[i]
        filename = article_to_str(i + 1)
        pathname = "./{}/{}".format(dirname, filename)

        if VERBOSE: print("[{}] {}".format(i+1,url.split('/')[-1]))

        # saves webpage to file
        urlretrieve(url, pathname)


############### PREPROCESSING FUNCTIONS ###############


def html_to_dict(article_n):
    # returns a dictionary mapping each attribute to its value
    page_dir = article_to_page_str(article_n)
    article_html = article_to_str(article_n)
    pathname = page_dir + "/" + article_html

    attr_dict = {"title":"", "bookseries":"", "author":"", "rating_value":"",
                 "rating_count":"", "review_count":"", "plot":"", "pages":"",
                 "publishing_date":"", "characters":"", "places":""}
   
    soup = soup_from_filename(pathname)
    
    for key in attr_dict.keys():
        try:
            attribute = get_attribute(soup, key)
            attr_dict[key] = attribute.replace("\t", " ").replace("\n", " ")
        except:
            error = "[{}] {} not found".format(article_n, key)
            if VERBOSE: print(error)

    try:
        if detect(attr_dict['plot']) != 'en':
            error = "[{}] Not in english!".format(article_n)

            if VERBOSE:
                print(error)
                return None
    except Exception as e:
        return None

    return attr_dict

def get_tokenizer():
    return RegexpTokenizer(r"[a-zA-Z]+|[0-9]+")

def preprocess(text):
    # returns the text after clean up
    to_remove = list(string.punctuation) +\
                stopwords.words('english') + ["..."]
    tokenizer = get_tokenizer()
    words = tokenizer.tokenize(text.lower())
    filtered = [w for w in words if w not in to_remove]
    porter = PorterStemmer()
    stemmed = [porter.stem(w) for w in filtered]
    
    return " ".join(stemmed)

def as_text(row):
    # takes a dataframe row and returns it as a string to be used in the search engine
    return "{} {}".format(row.bookTitle, row.Plot)


############### SEARCH ENGINE FUNCTIONS ###############


def get_largest(heap, n):
    return list(zip(*heapq.nlargest(n, heap)))

def cosine_similarity(v1, v2):
    # given two vectors returns its cosine similarity
    v1_norm = np.sqrt((v1**2).sum())
    v2_norm = np.sqrt((v2**2).sum())

    return v1.dot(v2) / (v1_norm * v2_norm)

def is_valid_series(series):
    # check if series is in the right format (#N)
    if re.search(r'[0-9][-|–][0-9]', series):
        return False
    else:
        return True

def first_series_cumulative_page_count():
    # plot the cumulative page count of the first 10 book series sorted by year
    df = load_dataset()
    cols = ['bookSeries','NumberofPages','PublishingDate']
    not_na = df.bookSeries.notna() & df.PublishingDate.notna()
    topseries = df[not_na][cols].groupby('bookSeries').head()
    filtered = topseries.bookSeries.apply(is_valid_series)
    topseries = topseries[filtered]
    topseries['Year'] = topseries['PublishingDate'].apply(lambda x: x.year)
    topseries['bookSeries'] = topseries['bookSeries'].apply(lambda x: re.sub(r'\s#[0-9]+','',x))
    topseries = topseries.drop('PublishingDate', axis=1)
    cols = ['bookSeries','NumberofPages','Year']
    first_bookseries = topseries[cols].sort_values(['Year']).head(10).bookSeries.tolist()
    topseries = topseries[topseries.bookSeries.isin(first_bookseries)].sort_values(['Year'])
    topseries['Year_std'] = topseries['Year'] - topseries['Year'].min()
    topseries['cumsum_pages'] = topseries['NumberofPages'].cumsum()
    plot = topseries.plot(x='Year_std',
                          y='cumsum_pages',
                          kind='line',
                          marker='o',
                          markersize=5,
                          legend=False,
                          title='First 10 books series cumulative page count by year')
    plot.set_xlabel('Year Count')
    plot.set_ylabel('Cumulative Page Count')
    plot = plot.set_xticklabels(topseries['Year'].unique())

def load_dataset():
    # returns the local dataset as a DataFrame
    books = pd.read_csv(TSV_FILENAME, sep='\t')
    books['ratingCount'] = books.ratingCount.replace(np.nan, 0)
    books['ratingCount'] = books.ratingCount.replace(',', '', regex=True).astype(int)
    books['PublishingDate'] = books['PublishingDate'].apply(to_date)
    books['NumberofPages'].fillna(0, inplace=True)

    return books

class SearchEngine:
    """
    A search engine class to manage the local file to make query efficient (frequency and tfidf)
    as well as different kind of query functions
    """

    def __init__(self):
        # load dataset
        try:
            self.books_df = load_dataset()
        except FileNotFoundError:
            if VERBOSE: print("Can't find ", TSV_FILENAME)
            return

        # load vocabulary or create it if doesn't exist
        try:
            self.load_vocabulary()
        except:
            if VERBOSE:
                print("Vocabulary not found")
                print("Trying to create it...")
            self.create_vocabulary()

            if VERBOSE: print("Vocabulary correctly generated")

        # load index or create it if doesn't exist
        try:
            self.load_index()
        except:
            if VERBOSE:
                print("Can't find {} or {}".format(INDEX_FILENAME, TFIDF_FILENAME))
                print("Trying to generate indexes")

            self.create_index()

            if VERBOSE:
                print("Index correctly generated")

    def create_vocabulary(self):
        # create the vocabulary file as {word: id}
        books = load_dataset()
        vocabulary = []

        for i,row in books.iterrows():
            vocabulary += preprocess(as_text(row)).split()

        # keep unique words and sort
        vocabulary = sorted(list(set(vocabulary)))
        vocabulary = dict(zip(vocabulary, range(len(vocabulary))))

        self.vocabulary = vocabulary

        # save vocabulary to file as index,word
        with open(vocabulary_filename,"wb") as f:
            pickle.dump(vocabulary,f)

    def create_index(self):
        # create the word index and saves it to file as word_id,doc_id1,doc_id2,...
        vocabulary = self.get_vocabulary()

        books = load_dataset()
        index = {} # term_id : book_id
        tfidf_index = {} # term_id : [doc_id, tfidf]
        book_size = {} # doc_id: number of words
        n_books = len(vocabulary.keys()) # number of books

        # compute the number of words of each book as well as
        # saving the documents ids in index[word]
        for i, row in books.iterrows():
            text = preprocess(as_text(row))
            words = text.split()

            for word in words:
                index.setdefault(vocabulary[word], []).append(i)

            book_size[i] = len(words)

        # compute both the index and tfidf for each word in the corpus
        for word_id, docs_ids in index.items():
            counter = Counter(docs_ids)

            # compute both the index and tfidf
            for doc_id, freq in counter.items():
                tf = freq / book_size[doc_id]
                idf = log(n_books / len(docs_ids))
                counter[doc_id] = tf * idf

            tfidf_index[word_id] = dict(counter)

            index[word_id] = list(counter.keys())

        with open(TFIDF_FILENAME,'wb') as f:
            pickle.dump(tfidf_index,f)

        with open(INDEX_FILENAME,'wb') as f:
            pickle.dump(index,f)

        self.index = index
        self.tfidf_index = tfidf_index
    
    def load_vocabulary(self):
        # returns the vocabulary as a dictionary {word:index}
        with open(vocabulary_filename,"rb") as f:
            self.vocabulary = pickle.load(f)

    def load_index(self):
        # return index as a dictionary {word_id:[doc_id1, doc_id2, ...]}
        # values are stored as int
        with open(INDEX_FILENAME,'rb') as f:
            self.index = pickle.load(f)

        with open(TFIDF_FILENAME,'rb') as f:
            self.tfidf_index = pickle.load(f)

    def get_vocabulary(self):
        return self.vocabulary

    def get_index(self,tfidf=False):
        if tfidf:
            return self.tfidf_index
        else:
            return self.index

    def vocabulary_size(self):
        return len(self.vocabulary)

    def tfidf_from_str(self, query):
        # takes a query and return its normalized tfidf vector representation
        n_books = len(self.books_df)
        size = len(self.vocabulary.keys())
        vector = np.zeros(size)

        terms = preprocess(query).split()

        for term in terms:
            term_id = self.vocabulary[term]
            frequency = len(self.index[self.vocabulary[term]])
            vector[term_id] = 1/len(terms)*log(n_books/frequency)

        return vector/vector.sum()

    def tfidf(self, term_id, doc_id):
        # return the tfidf score of term_id in doc_id
        try:
            return self.tfidf_index[term_id][doc_id]
        except:
            return 0

    def tfidf_vector(self, doc_id):
        # takes a doc_id and returns the vector representation of the doc w tfidf
        terms = preprocess(self.books_df.iloc[doc_id,:].Plot).split()
        vector = np.zeros(self.vocabulary_size())

        for term in terms:
            term_id = self.vocabulary[term]
            vector[term_id] = self.tfidf(term_id,doc_id)

        return vector/vector.sum()

    def query(self, query, type_='and', n_results=N_RESULTS):
        # type can be ['and','cosine']
        if type_== 'and':
            result_ids = self.and_query(query)
            return self.books_df.iloc[result_ids][DEFAULT_COLS].head(n_results)
        elif type_ == 'cosine':
            return self.similarity_query(query, n_results=n_results)
        elif type_ == 'rating':
            return self.rating_query(query, n_results=n_results)
        elif type_ == 'title':
            return self.title_query(query, n_results=n_results)
        else:
            print("Invalid query type")

    def and_query(self, query):
        """
        Given a query returns a DataFrame of documents where each word
        of the query appears in either the title or plot
        if visualize is False just return the documents ids
        """
        query_words = preprocess(query).split()
        documents_ids = []
        result_ids = set(self.index[self.vocabulary[query_words[0]]])

        # compute intersection of documents containing each word in the query
        for word in query_words[1:]:
            documents = self.index[self.vocabulary[word]]
            result_ids = list(result_ids.intersection(frozenset(documents)))

        return list(result_ids)

    def similarity_query(self, query, n_results=N_RESULTS):
        """
        return the top n_results according to the cosine similarity between
        the plot+title ande the query
        """
        scores = {}
        books = self.books_df
        size = self.vocabulary_size()
        query_tfidf = self.tfidf_from_str(query)
        h = []

        # only parse documents that contain all query words
        docs_ids = self.and_query(query)

        # compute similarity score and save the results in a heap structure
        for doc_id in docs_ids:
            score = self.tfidf_vector(doc_id)
            sim = cosine_similarity(score, query_tfidf)
            heapq.heappush(h, (sim, doc_id))

        # take the top n_results in the heap
        similarities, result_ids = get_largest(h, n_results)

        df = books.iloc[list(result_ids),:][DEFAULT_COLS]
        df['Similarity'] = np.round(np.array(similarities), 2)

        return df

    def rating_query(self, query, n_results=N_RESULTS):
        # return the top n_results according to their rating aggregate value
        scores = {}
        lengths = {}
        h = []
        size = self.vocabulary_size()
        query_tfidf = self.tfidf_from_str(query)
        books = self.books_df
        docs_ids = self.and_query(query)
        books['ratingValue'] /= books['ratingValue'].max()

        # sort documents by their rating aggregate value
        for doc_id in docs_ids:
            book = books.iloc[doc_id]

            score = self.tfidf_vector(doc_id)
            sim = cosine_similarity(score, query_tfidf)

            # aggregate rating value
            rating = book['ratingValue'] * sim

            # store in the heap
            heapq.heappush(h, (rating, doc_id))

        similarities, result_ids = get_largest(h, n_results)
        similarities = np.array(similarities)
        similarities /= similarities.max()

        df = books.iloc[list(result_ids),:][DEFAULT_COLS]
        df['Similarity'] = np.round(similarities, 2)

        return df

    def title_query(self, query, n_results=N_RESULTS):
        """
        return the top n_results according to their cosine similarity
        between the title and the query
        """
        scores = {}
        lengths = {}
        h = []
        size = self.vocabulary_size()
        books = self.books_df
        query_tfidf = self.tfidf_from_str(query)
        docs_ids = self.and_query(query)

        for doc_id in docs_ids:
            book = books.iloc[doc_id]
            title_vector = self.tfidf_from_str(book.bookTitle)
            title_sim = cosine_similarity(title_vector, query_tfidf)
            heapq.heappush(h, (title_sim, doc_id))

        similarities, result_ids = get_largest(h, n_results)

        df = books.iloc[list(result_ids),:][DEFAULT_COLS]
        similiarities = np.array(similarities)
        df['Similarity'] = np.round(similarities, 2)

        return df
