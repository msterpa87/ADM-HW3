# ADM-HW3

### Directory descriptions:

1. **`books`**:
> a list of url of the first 300 most read books on [GoodReads](https://www.goodreads.com/)
2. **`index`** and **`tfidf_index`**:
> pickle dictionaries representing the inverted index `{word_id: [docs_ids]}` and tfidf `{word_id: [doc_id:freq]}`
3. **` dataset.tsv`**:
> records of each book with each respective tag content appropriately parsed
4. **`functions.py`**:
> all functions used in the notebook and in the scripts
5. **`build_dataset.py`**:
> creates `dataset.tsv` from all documents contained in `page_[i]/` subdirectories named `article_[i].html`
