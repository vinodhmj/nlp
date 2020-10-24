# Web scraping, pickle imports
import requests
from bs4 import BeautifulSoup
import pickle
import re

# Scrapes transcript data from discoverypoetry.com
def url_to_urls():
    '''Returns individual links from https://discoverpoetry.com/poems/100-most-famous-poems/'''
    baseurl = "https://discoverpoetry.com/poems/100-most-famous-poems/"
    page = requests.get(baseurl).text
    soup = BeautifulSoup(page, "lxml")
    authors = []
    links = []
    title = []
    for tag in soup.find_all('a',{"class":"poem-listing-link"}):
        if tag.get('href') == "#":
            continue
        else:
            links.append(tag.get('href'))
            authors.append(tag.find(class_="poem-listing-author").text)
            title.append(tag.find(class_="poem-listing-title").text)
    return authors, links, title

# Scrapes transcript data from scrapsfromtheloft.com
def url_to_transcript(url):
    '''Returns transcript data specifically from all the links scraped from https://discoverpoetry.com/poems/100-most-famous-poems/'''
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find(class_="poem-text").find_all('p')]
    print(url)
    return text

def main():
    print("hello")
    result = url_to_urls()
    authors = result[0]
    urls = result[1]
    titles = result[2]
    
    transcripts = [url_to_transcript(u) for u in urls]

    # Pickle files for later use

    # Make a new directory to hold the text files  
    # !mkdir transcripts    
    
    for i, c in enumerate(authors):
        authors[i] = re.sub('[\.\,\—\?\!\']', '', c[3:])
        titles[i] = re.sub('[\.\,\—\?\!\']', '', titles[i])
        with open("transcripts/" + titles[i] + ".txt", "wb") as file:
            pickle.dump(transcripts[i], file)

    with open("transcripts/" + "authors" + ".txt", "wb") as file:
        pickle.dump(authors, file)

    with open("transcripts/" + "titles" + ".txt", "wb") as file:
        pickle.dump(titles, file)

    # Load pickled files
    authors = []
    with open("transcripts/" + "authors" + ".txt", "rb") as file:
        authors = pickle.load(file)

    titles = []
    with open("transcripts/" + "titles" + ".txt", "rb") as file:
        titles = pickle.load(file)

    print(authors)
    print(titles)
        
    data = {}
    for i, c in enumerate(titles):
        with open("transcripts/" + c + ".txt", "rb") as file:
            data[c] = pickle.load(file)

    print(data['Ozymandias'])




# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
    
