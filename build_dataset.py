from functions import *

# creates a tsv file from the html documents in the database
add_header = True
chunk = ""
book_list = bookslist_from_file()

if os.path.exists(TSV_FILENAME):
    add_header = False

with open(TSV_FILENAME, 'a', encoding = "utf-8") as f:
    if add_header: f.write("\t".join(HEADER_LIST) + "\n")

    for i in range(1, 301):
        try:
            attr_dict = html_to_dict(i)
        except:
            # download missing files
            if VERBOSE: print("[{}] File not found, download...".format(i))
            download_article(book_list, i)
            attr_dict = html_to_dict(i)

        # plot not in english
        if attr_dict == None:
            continue
        
        row_list = list(attr_dict.values()) + [book_list[i-1]]
        chunk += "\t".join(row_list) + "\n"

        if VERBOSE: print("[{}] OK".format(i))

        if i % CHUNK_SIZE == 0:
            f.write(chunk)
            chunk = ""