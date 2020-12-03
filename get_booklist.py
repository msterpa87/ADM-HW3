from functions import *

movies = []

with open(URL_FILENAME, 'w') as file:

    for i in range(MAX_PAGE):
        if i % 24 == 0 and VERBOSE:
            print("Scraping page: " + str(i + 1))

        url = BESTBOOK_URL + str(i+1)
        html = urlopen(url)
        soup = BeautifulSoup(html, 'lxml')

        for tag in soup.find_all('a', {'class': 'bookTitle'}):
            file.write(BASE_URL + tag['href'] + "\n")