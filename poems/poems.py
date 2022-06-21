# Web scraping, pickle imports
import requests
from bs4 import BeautifulSoup
import pickle
import re
import string
import pandas as pd
import io 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from wordcloud import WordCloud
import matplotlib.pyplot as plt

    
# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\r', ' ', text)
    text = re.sub('—', ' ', text)    
    return text

# We are going to change this to key: comedian, value: string format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    return text

# Scrapes transcript data from scrapsfromtheloft.com
def url_to_transcript(url):
    '''Returns transcript data specifically from all the links scraped from https://discoverpoetry.com/poems/100-most-famous-poems/'''
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find_all('p')]
    print(url)
    return text

def main():
    print("hello")
    urls = ["https://en.wikipedia.org/wiki/Electric_aircraft", "https://edition.cnn.com/travel/article/electric-aircraft/index.html",
            "https://www.cnbc.com/2021/11/19/rolls-royce-says-its-electric-aircraft-hit-a-top-speed-of-387-mph.html",
            "https://www.rollingstone.com/culture/culture-features/electric-airplanes-climate-change-solar-power-1323576/",
            "https://www.weforum.org/agenda/2020/11/electric-planes-aviation-future-innovation/",
            "https://www.airbus.com/en/innovation/zero-emission/electric-flight",
            "https://www.wired.com/2016/06/nasas-new-electric-plane-looks-goofy-packs-sweet-tech/",
            "https://www.wired.com/2015/01/electric-airplanes-future-pilot-training/",
            "https://www.wired.com/2014/07/chip-yates-electric-plane-records/",
            "https://www.forbes.com/sites/jeremybogaisky/2021/02/18/hybrid-electric-aviation-pioneer-ampaire-to-be-acquired-by-surf-air/?sh=55b853a575a6",
            "https://ampaireinc.medium.com/on-becoming-a-2021-ted-fellow-an-opportunity-to-spread-the-word-about-electric-aviation-7d02833dd0fa",
            "https://www.businessinsider.com/electric-planes-future-of-aviation-problems-regulations-2020-3?r=US&IR=T",
            "https://www.economist.com/technology-quarterly/2019/05/30/smaller-planes-could-soon-use-electric-propulsion",
            "https://www.ainonline.com/aviation-news/business-aviation/2020-10-12/ampaires-electric-eel-skymaster-makes-longest-flight-yet",
            "https://www.nbcnews.com/science/science-news/largest-electric-plane-yet-completed-its-first-flight-it-s-n1221401",
            "https://www.mordorintelligence.com/industry-reports/more-electric-aircraft-market"]


    # transcripts = [url_to_transcript(u) for u in urls]
    # print(transcripts)
    
    # with open("transcripts/" + "electric" + ".txt", "wb") as file:
    #     pickle.dump(transcripts, file)

    pickledData = []
    with open("transcripts/" + "electric" + ".txt", "rb") as file:
        pickledData = pickle.load(file)

    combined_text = ''   
    for data in pickledData:
        combined_text = combined_text + ' '.join(data)

    data = [combined_text]    
    df = pd.DataFrame(data)
    df.columns = ['transcript']

    # Apply a first round of text cleaning techniques
    round1 = lambda x: clean_text_round1(x)

    # Let's take a look at the updated text
    data_clean = pd.DataFrame(df.transcript.apply(round1))
    print(data_clean)


    round2 = lambda x: clean_text_round2(x)
    # Let's take a look at the updated text
    data_clean = pd.DataFrame(data_clean.transcript.apply(round2))
    print(data_clean)

    # Let's pickle it for later use
    df.to_pickle("electricCleanData.pkl")

    # We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words


    cv = CountVectorizer(stop_words='english')
    data_cv = cv.fit_transform(data_clean.transcript)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = data_clean.index
    print(data_dtm)


    data_dtm.to_pickle("electricdtm.pkl")

    # Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
    data_clean.to_pickle('electric_data_clean.pkl')
    pickle.dump(cv, open("electriccv.pkl", "wb"))


    data = pd.read_pickle('electricdtm.pkl')
    data = data.transpose()

    data_clean = pd.read_pickle('electric_data_clean.pkl')
    print(data_clean)
    # print(data_clean.transcript)

    data_combined = combine_text(data_clean.transcript)
    # print(data_combined)

    # Let's update our document-term matrix with the new list of stop words

    # Read in cleaned data
    data_clean = pd.read_pickle('data_clean.pkl')

    # Add new stop words
    stop_words = text.ENGLISH_STOP_WORDS
    #add words that aren't in the NLTK stopwords list
    new_stopwords = ['lot', 'flight', 'S', 'plane', 'faa', 'shahani', 't', 'called', 'took', 'way', 'longer', 'known', 'electric aircraft', 'say', 'magnix', 'just', 'saved', 'use', 'flights']
    new_stopwords_list = stop_words.union(new_stopwords)
    not_stopwords = []
    final_stop_words = set(new_stopwords_list)

    
    # Recreate document-term matrix
    cv = CountVectorizer(stop_words=final_stop_words)
    data_cv = cv.fit_transform(data_clean.transcript)
    data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_stop.index = data_clean.index

    pickle.dump(cv, open("electric_cv_stop.pkl", "wb"))
    data_stop.to_pickle("electric_dtm_stop.pkl")

    wc = WordCloud(width = 1920, height = 1080, stopwords=final_stop_words, background_color="white", colormap="Dark2", max_font_size=150, random_state=85)

    
    sText = re.sub('like', '', data_combined)
    sText = data_combined
    wc.generate(sText)

    # Reset the output dimensions
    with open("Output.svg", "w") as text_file:
        text_file.write(wc.to_svg())

    plt.figure(figsize = (16,9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    plt.show()

    # This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
    
