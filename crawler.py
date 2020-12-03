from functions import *
import threading
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_threads', type=int, default=10)
	parser.add_argument('--start', type=int, default=1)
	parser.add_argument('--end', type=int, default=300000)
	args = parser.parse_args()

	return args


if __name__ == "__main__":
	args = get_args()
	start, end, n_threads = args.start, args.end, args.n_threads
	book_list = bookslist_from_file()

	# group article to assign to each thread
	size = (end - start) // n_threads
	bins = list(range(start - 1, end + 1, size))

	if n_threads == 1: bins = [start - 1, end]

	threads = []

	for i in range(n_threads):
	    t = threading.Thread(target = crawler, 
	                         args = (book_list, bins[i], bins[i+1]))
	    threads.append(t)
	    t.start()

	for t in threads: t.join()