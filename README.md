# ADM-HW3

### Usage:
- `python get_booklist.py`: download urls and saves them into the file `books`
- `python crawler.py [--n_threads N_THREADS] [--start START] [--end END]`: starts `N_THREADS` threads to download from `START` to `END` books taking url from the file `books`
- `python build_dataset.py`: reads the html files preprocessing their content and saves it into a `dataset.tsv` file

### Directory descriptions:

1. **`books`**:
> a list of url of the first 300 most read books on [GoodReads](https://www.goodreads.com/)
2. **`index`** and **`tfidf_index`**:
> pickle dictionaries representing the inverted index `{word_id: [docs_ids]}` and tfidf `{word_id: [doc_id:freq]}`
3. **`dataset.tsv`**:
> records of each book with each respective tag content appropriately parsed
4. **`functions.py`**:
> all functions used in the notebook and in the scripts
5. **`get_booklist.py`**:
> download the urls of the first 300 pages of most read books from GoodReads
6. **`crawler.py`**:
> downloads the urls in `books` and saves the content in html files as `article_[i].html` organizing them into directors `page_[i]/`
7. **`build_dataset.py`**:
> creates `dataset.tsv` from all documents contained in `page_[i]/` subdirectories named `article_[i].html`
8. **`longest_subsequence.py`**:
> contains both the recursive and dynamic programming solution to the longest increasing subsequence problem
